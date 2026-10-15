//! Private openings over YPIR's SimplePIR variant.
//!
//! The PIR database has one row per pattern. Row `r` contains the records
//! in `PrivateLayout::row(r)`: each slot contributes its complete 14-bit
//! symbols, followed by a shared suffix packing all slots' residual bits.
//! Padding occurs only at the end of the row. A maintainer keeps every
//! record once and never builds the rows:
//! - per query, it adds the query coefficients of all patterns that place a
//!   record in a slot, then multiplies the sum with the stored record;
//! - its preprocessing gathers adjacent database columns in small batches
//!   from the stored records.
//!
//! Both give exactly what YPIR computes on the table of all rows, so YPIR's
//! response packing, client, and decoder apply unchanged.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::arith::{barrett_coeff_u64, barrett_u64};
use spiral_rs::client::Client;
use spiral_rs::params::Params;
use spiral_rs::poly::{PolyMatrix, PolyMatrixNTT};
use ypir::factored;
use ypir::params::{GetQPrime, params_for_scenario_simplepir};
use ypir::scheme::SEED_0;
use ypir::server::{OfflinePrecomputedValues, YServer};

use crate::Fp;
use crate::private::layout::PrivateLayout;
use crate::private::records::record_entries;
use crate::unauth::UnauthStore;
use crate::value::{decode_value, encode_value};

/// Bits of one plaintext cell of the PIR database.
pub const CELL_BITS: usize = 14;
/// Bytes of one stored entry.
pub const ENTRY_BYTES: usize = 32;

const LOW_MASK: u64 = (1u64 << 32) - 1;
/// Marks an empty slot in the row table.
const NO_RECORD: u32 = u32::MAX;
/// Adjacent columns gathered together for the initial hint. Each worker
/// keeps at most this many columns, independently of the database width.
pub const HINT_COLUMN_BATCH: usize = 1024;
// Shared across hint workers, excluding cache-line padding and the minimum
// one column per worker. The database itself is never materialized.
const HINT_GATHER_BYTES: usize = 128 * 1024 * 1024;
// One cache line between temporary columns prevents power-of-two row
// counts from mapping every column's current tile to the same cache sets.
const HINT_COLUMN_PADDING: usize = 32;

/// Plaintext cells of one stored entry, the width of a slot above the
/// group's subtree.
pub const UPPER_CELLS: usize = (ENTRY_BYTES * 8).div_ceil(CELL_BITS);

/// Dimensions of the PIR database of a layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PirShape {
    /// Rows that hold a pattern.
    pub patterns: usize,
    /// Slots per record: the groups of the pattern's subtree and one entry
    /// per level above it.
    pub slots: usize,
    /// Slots that hold groups, the first slots of a record.
    pub group_slots: usize,
    /// Payload bits per group, excluding padding in its last stored cell.
    pub record_bits: usize,
    /// Stored cells per group: `ceil(record_bits / CELL_BITS)`.
    pub record_cells: usize,
}

impl PirShape {
    /// Shape of the database of `layout`.
    pub fn of(layout: &PrivateLayout) -> Self {
        Self {
            patterns: layout.patterns(),
            slots: layout.slots(),
            group_slots: layout.group_slots(),
            record_bits: layout.entries_per_record() * ENTRY_BYTES * 8,
            record_cells: (layout.entries_per_record() * ENTRY_BYTES * 8).div_ceil(CELL_BITS),
        }
    }

    /// Slots above the pattern's subtree.
    pub fn upper_slots(&self) -> usize {
        self.slots - self.group_slots
    }

    /// Plaintext cells of one slot.
    pub fn slot_width(&self, slot: usize) -> usize {
        if slot < self.group_slots {
            self.record_cells
        } else {
            UPPER_CELLS
        }
    }

    /// Number of payload bits in a slot; stored last-cell padding is excluded.
    pub fn slot_bits(&self, slot: usize) -> usize {
        if slot < self.group_slots {
            self.record_bits
        } else {
            ENTRY_BYTES * 8
        }
    }

    /// First full symbol of a slot. Its residual bits live in the shared
    /// suffix instead of occupying a separately padded symbol here.
    pub fn slot_offset(&self, slot: usize) -> usize {
        if slot < self.group_slots {
            slot * (self.record_bits / CELL_BITS)
        } else {
            self.group_slots * (self.record_bits / CELL_BITS)
                + (slot - self.group_slots) * (ENTRY_BYTES * 8 / CELL_BITS)
        }
    }

    fn full_columns(&self) -> usize {
        self.slot_offset(self.slots)
    }

    /// Bit position of a slot's residual bits in the shared suffix.
    fn tail_bit_offset(&self, slot: usize) -> usize {
        if slot < self.group_slots {
            slot * (self.record_bits % CELL_BITS)
        } else {
            self.group_slots * (self.record_bits % CELL_BITS)
                + (slot - self.group_slots) * (ENTRY_BYTES * 8 % CELL_BITS)
        }
    }

    fn packing(&self, slot: usize) -> SlotPacking {
        SlotPacking::new(
            self.slot_offset(slot),
            self.slot_bits(slot),
            self.full_columns(),
            self.tail_bit_offset(slot),
        )
    }

    /// Plaintext symbols of the row, rounding once after all payload bits.
    pub fn row_cells(&self) -> usize {
        self.full_columns() + self.tail_bit_offset(self.slots).div_ceil(CELL_BITS)
    }

    /// YPIR parameters for this shape, shared by all users of the shape.
    pub fn params(&self) -> &'static Params {
        static CACHE: OnceLock<Mutex<HashMap<(usize, usize), &'static Params>>> = OnceLock::new();
        // The key is the backend's padding of the shape, computed before any
        // `Params` is built: building one also builds its NTT tables.
        const POLY_LEN_LOG2: usize = 11;
        let row_bits = self.row_cells() * CELL_BITS;
        let key = (
            (self.patterns.next_power_of_two().trailing_zeros() as usize)
                .saturating_sub(POLY_LEN_LOG2),
            (row_bits as f64 / ((1 << POLY_LEN_LOG2) * CELL_BITS) as f64).ceil() as usize,
        );
        let mut cache = CACHE
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("PIR parameter cache");
        *cache.entry(key).or_insert_with(|| {
            let candidate = params_for_scenario_simplepir(self.patterns, row_bits);
            assert_eq!(key, (candidate.db_dim_1, candidate.instances));
            Box::leak(Box::new(candidate))
        })
    }

    /// Rows of the PIR database, including padding rows.
    pub fn db_rows(&self) -> usize {
        factored::simplepir_db_rows(self.params())
    }

    /// Columns of the PIR database, including padding columns.
    pub fn db_cols(&self) -> usize {
        factored::simplepir_db_cols(self.params())
    }

    /// Ciphertexts of a response: one per `poly_len` database columns.
    pub fn outputs(&self) -> usize {
        let params = self.params();
        factored::simplepir_db_cols(params) / params.poly_len
    }

    /// Words of the packing keys a client sends with its query.
    pub fn pack_key_words(&self) -> usize {
        pack_key_words(self.params())
    }

    /// Bytes of one ciphertext of a response.
    pub fn ciphertext_bytes(&self) -> usize {
        let params = self.params();
        let bits = (params.get_q_prime_1() as f64).log2().ceil() as usize
            + (params.get_q_prime_2() as f64).log2().ceil() as usize;
        (bits * params.poly_len).div_ceil(8)
    }
}

/// Placement of one slot's full symbols and residual bits. Suffix
/// contributions use disjoint bits, so adding them preserves PIR linearity.
#[derive(Clone, Copy)]
struct SlotPacking {
    base: usize,
    full: usize,
    tail_base: usize,
    tail_bits: usize,
    shift: usize,
}

impl SlotPacking {
    fn new(base: usize, bits: usize, tail_base: usize, tail_offset: usize) -> Self {
        Self {
            base,
            full: bits / CELL_BITS,
            tail_base: tail_base + tail_offset / CELL_BITS,
            tail_bits: bits % CELL_BITS,
            shift: tail_offset % CELL_BITS,
        }
    }

    fn tail_columns(&self) -> usize {
        if self.tail_bits == 0 {
            0
        } else {
            (self.shift + self.tail_bits).div_ceil(CELL_BITS)
        }
    }

    fn columns(&self) -> usize {
        self.full + self.tail_columns()
    }

    fn column(&self, offset: usize) -> usize {
        if offset < self.full {
            self.base + offset
        } else {
            self.tail_base + offset - self.full
        }
    }

    #[inline]
    fn cell(&self, cells: &[u16], offset: usize) -> u16 {
        if offset < self.full {
            return cells[offset];
        }
        let value = (cells[self.full] & ((1 << self.tail_bits) - 1)) as u32;
        (((value << self.shift) >> ((offset - self.full) * CELL_BITS)) & ((1 << CELL_BITS) - 1))
            as u16
    }

    /// Restore the separately packed slot format expected by the opening
    /// decoder. Only the final partial symbol needs bit extraction.
    fn unpack(&self, row: &[u64]) -> Vec<u64> {
        let mut cells = Vec::with_capacity(self.full + usize::from(self.tail_bits != 0));
        cells.extend_from_slice(&row[self.base..self.base + self.full]);
        if self.tail_bits != 0 {
            let mut value = row[self.tail_base] >> self.shift;
            if self.tail_columns() == 2 {
                value |= row[self.tail_base + 1] << (CELL_BITS - self.shift);
            }
            cells.push(value & ((1 << self.tail_bits) - 1));
        }
        cells
    }
}

/// Write the entries of a block as plaintext cells, least significant bits
/// first.
pub fn entries_to_cells(entries: &[Fp], out: &mut [u16]) {
    bytes_to_cells(entries.iter().flat_map(|entry| encode_value(*entry)), out)
}

/// Write a byte string as plaintext cells, least significant bits first.
pub fn bytes_to_cells(bytes: impl IntoIterator<Item = u8>, out: &mut [u16]) {
    let mut acc = 0u64;
    let mut acc_bits = 0usize;
    let mut next = 0usize;
    let mask = (1u64 << CELL_BITS) - 1;
    {
        for byte in bytes {
            acc |= (byte as u64) << acc_bits;
            acc_bits += 8;
            while acc_bits >= CELL_BITS {
                out[next] = (acc & mask) as u16;
                next += 1;
                acc >>= CELL_BITS;
                acc_bits -= CELL_BITS;
            }
        }
    }
    if acc_bits > 0 {
        out[next] = acc as u16;
        next += 1;
    }
    out[next..].fill(0);
}

/// Write entry `entry_in_record` into the cells of a record whose bits for
/// that entry are zero. Writing every entry of a record this way gives the
/// cells of [`entries_to_cells`].
pub fn or_entry_into_cells(cells: &mut [u16], entry_in_record: usize, entry: Fp) {
    let mask = (1u16 << CELL_BITS) - 1;
    let first_bit = entry_in_record * ENTRY_BYTES * 8;
    for (k, byte) in encode_value(entry).into_iter().enumerate() {
        let bit = first_bit + 8 * k;
        let cell = bit / CELL_BITS;
        let shift = bit % CELL_BITS;
        cells[cell] |= ((byte as u16) << shift) & mask;
        if shift + 8 > CELL_BITS {
            cells[cell + 1] |= (byte as u16) >> (CELL_BITS - shift);
        }
    }
}

/// Read entry `entry_in_record` back from the cells of a block.
pub fn entry_from_cells(cells: &[u64], entry_in_record: usize) -> Fp {
    decode_value(bytes_from_cells(cells, entry_in_record))
}

/// Read the bytes of entry `entry_in_record` from the cells of a block.
pub fn bytes_from_cells(cells: &[u64], entry_in_record: usize) -> [u8; ENTRY_BYTES] {
    let mut bytes = [0u8; ENTRY_BYTES];
    let first_bit = entry_in_record * ENTRY_BYTES * 8;
    for (k, byte) in bytes.iter_mut().enumerate() {
        let bit = first_bit + 8 * k;
        let cell = bit / CELL_BITS;
        let shift = bit % CELL_BITS;
        let mut value = cells[cell] >> shift;
        if shift + 8 > CELL_BITS {
            value |= cells[cell + 1] << (CELL_BITS - shift);
        }
        *byte = value as u8;
    }
    bytes
}

/// The packing keys of a client, as plain words: for each of the
/// `poly_len_log2` keys, `t_exp_left` polynomials of `poly_len` words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackKeys {
    pub words: Vec<u64>,
}

impl PackKeys {
    fn from_matrices(params: &Params, matrices: &[PolyMatrixNTT<'static>]) -> Self {
        let mut words = Vec::with_capacity(pack_key_words(params));
        for matrix in matrices {
            assert_eq!((matrix.rows, matrix.cols), (1, params.t_exp_left));
            for col in 0..matrix.cols {
                // A condensed polynomial keeps its values in the first
                // `poly_len` words.
                words.extend_from_slice(&matrix.get_poly(0, col)[..params.poly_len]);
            }
        }
        Self { words }
    }

    fn to_matrices(
        &self,
        params: &'static Params,
    ) -> Result<Vec<PolyMatrixNTT<'static>>, PirError> {
        if self.words.len() != pack_key_words(params) {
            return Err(PirError::PackKeys);
        }
        let per_matrix = params.t_exp_left * params.poly_len;
        Ok(self
            .words
            .chunks_exact(per_matrix)
            .map(|chunk| {
                let mut matrix = PolyMatrixNTT::zero(params, 1, params.t_exp_left);
                for (col, poly) in chunk.chunks_exact(params.poly_len).enumerate() {
                    matrix.get_poly_mut(0, col)[..params.poly_len].copy_from_slice(poly);
                }
                matrix
            })
            .collect())
    }
}

/// Words of the packing keys for `params`.
fn pack_key_words(params: &Params) -> usize {
    params.poly_len_log2 * params.t_exp_left * params.poly_len
}

/// A request or a response that does not fit the database shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PirError {
    /// The query does not have one coefficient per database row.
    Query,
    /// The packing keys do not have the expected size.
    PackKeys,
    /// The response does not have the expected size.
    Response,
}

impl std::fmt::Display for PirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PirError::Query => write!(f, "query length does not match the database"),
            PirError::PackKeys => write!(f, "packing keys do not match the database"),
            PirError::Response => write!(f, "response size does not match the database"),
        }
    }
}

impl std::error::Error for PirError {}

/// Time of the two steps of a preprocessing.
#[derive(Clone, Copy, Debug)]
pub struct PreprocessTimes {
    /// The row table and the cells of the upper nodes, read from the
    /// records before the hint. Zero for a database whose rows are written
    /// out in full before the hint.
    pub gather: std::time::Duration,
    /// The SimplePIR hint, read column by column from the records.
    pub hint: std::time::Duration,
    /// The response-packing tables derived from the hint.
    pub packing: std::time::Duration,
}

/// Time of the two steps of a preprocessing update.
#[derive(Clone, Copy, Debug)]
pub struct UpdateTimes {
    /// Reading the changed records and patching the hint.
    pub hint: std::time::Duration,
    /// The response-packing tables derived from the patched hint.
    pub packing: std::time::Duration,
}

/// The same record IDs viewed either by pattern (stored state and small
/// samples) or by slot (a transient view shared by full-build workers).
#[derive(Clone, Copy)]
struct HintRowIds<'a> {
    values: &'a [u32],
    pattern_stride: usize,
    slot_stride: usize,
}

impl<'a> HintRowIds<'a> {
    fn by_pattern(values: &'a [u32], shape: &PirShape) -> Self {
        assert_eq!(values.len(), shape.patterns * shape.slots);
        Self {
            values,
            pattern_stride: shape.slots,
            slot_stride: 1,
        }
    }

    fn by_slot(values: &'a [u32], shape: &PirShape) -> Self {
        assert_eq!(values.len(), shape.patterns * shape.slots);
        Self {
            values,
            pattern_stride: 1,
            slot_stride: shape.patterns,
        }
    }
}

/// The records of one claim as a PIR database, with its preprocessing.
#[derive(Clone)]
pub struct PirDatabase {
    shape: PirShape,
    params: &'static Params,
    /// `records x record_cells` plaintext cells of the blocks.
    cells: Vec<u16>,
    records: usize,
    /// Per level above the groups, `count x UPPER_CELLS` plaintext cells:
    /// the entry of every stored node of that level, by position.
    upper: Vec<Vec<u16>>,
    /// `patterns x slots` ids; [`NO_RECORD`] for an empty slot. The first
    /// `group_slots` slots hold group numbers; the remaining slots hold
    /// positions of upper nodes within their level.
    rows: Vec<u32>,
    /// The SimplePIR hint, column-major: `hint_cols[col * poly_len + i]`,
    /// canonical modulo the ciphertext modulus. A patch of one column is a
    /// contiguous range; a growing layout can relocate the residual suffix.
    hint_cols: Vec<u64>,
    /// The response-packing tables of the hint; its own hint is empty.
    offline: OfflinePrecomputedValues<'static>,
    /// Threads that derive the packing tables of an update.
    table_threads: usize,
}

impl PirDatabase {
    /// Build the database of claim `registry_index` from its stored primary
    /// columns and preprocess it. The columns of the hint and the tables of
    /// the reply ciphertexts are independent and are computed on every
    /// available core, now and at every update.
    pub fn build(store: &UnauthStore, layout: &PrivateLayout, registry_index: usize) -> Self {
        let shape = PirShape::of(layout);
        let records = layout.records();
        let mut cells = vec![0u16; records * shape.record_cells];
        for record in 0..records {
            let entries = record_entries(store, layout, registry_index, record);
            entries_to_cells(
                &entries,
                &mut cells[record * shape.record_cells..(record + 1) * shape.record_cells],
            );
        }
        // One thread per block of columns of a reply ciphertext at most: a
        // small database gains nothing from more, and every thread's
        // allocator keeps what the preprocessing freed.
        let outputs = factored::simplepir_db_cols(shape.params()) / shape.params().poly_len;
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .clamp(1, outputs.max(1));
        let (mut database, _) = Self::from_cells_timed(layout, cells, threads);
        database.set_table_threads(threads);
        database
    }

    /// Build the database from the primary-column entries of all segments,
    /// `kappa` per segment in order.
    pub fn from_entries(layout: &PrivateLayout, entries: &[Fp]) -> Self {
        Self::from_cells(layout, Self::cells_from_entries(layout, entries))
    }

    /// The plaintext cells of all records, from the primary-column entries
    /// of all segments.
    pub fn cells_from_entries(layout: &PrivateLayout, entries: &[Fp]) -> Vec<u16> {
        let shape = PirShape::of(layout);
        let per_record = layout.entries_per_record();
        let records = layout.records();
        let mut cells = vec![0u16; records * shape.record_cells];
        let mut padded = vec![crate::field::fp_zero(); per_record];
        for record in 0..records {
            let start = (record * per_record).min(entries.len());
            let end = ((record + 1) * per_record).min(entries.len());
            padded[..end - start].copy_from_slice(&entries[start..end]);
            padded[end - start..].fill(crate::field::fp_zero());
            entries_to_cells(
                &padded,
                &mut cells[record * shape.record_cells..(record + 1) * shape.record_cells],
            );
        }
        cells
    }

    /// Build the database from the plaintext cells of all blocks.
    pub fn from_cells(layout: &PrivateLayout, cells: Vec<u16>) -> Self {
        Self::from_cells_timed(layout, cells, 1).0
    }

    /// The cells of every upper node, level by level, read from the cells
    /// of the block that holds its entry.
    pub fn upper_from_cells(layout: &PrivateLayout, cells: &[u16]) -> Vec<Vec<u16>> {
        let shape = PirShape::of(layout);
        let per_record = layout.entries_per_record();
        let mut block = vec![0u64; shape.record_cells];
        let mut last_record = usize::MAX;
        (0..layout.upper_slots())
            .map(|level| {
                let count = layout.upper_count(level);
                let mut upper = vec![0u16; count * UPPER_CELLS];
                for pos in 0..count {
                    let index = layout.upper_entry_index(level, pos);
                    let record = index / per_record;
                    if record != last_record {
                        for (wide, &cell) in block.iter_mut().zip(
                            &cells[record * shape.record_cells..(record + 1) * shape.record_cells],
                        ) {
                            *wide = cell as u64;
                        }
                        last_record = record;
                    }
                    bytes_to_cells(
                        bytes_from_cells(&block, index % per_record),
                        &mut upper[pos * UPPER_CELLS..(pos + 1) * UPPER_CELLS],
                    );
                }
                upper
            })
            .collect()
    }

    /// Derive the packing tables of later updates on `threads` threads,
    /// one output per thread at a time.
    pub fn set_table_threads(&mut self, threads: usize) {
        self.table_threads = threads.max(1);
    }

    /// [`Self::from_cells`] with the hint and the tables computed on
    /// `threads` threads, returning the time of the two preprocessing steps.
    pub fn from_cells_timed(
        layout: &PrivateLayout,
        cells: Vec<u16>,
        threads: usize,
    ) -> (Self, PreprocessTimes) {
        let shape = PirShape::of(layout);
        let records = layout.records();
        assert_eq!(cells.len(), records * shape.record_cells);
        let params = shape.params();
        let start = std::time::Instant::now();
        let rows = layout.rows_table();
        let upper = Self::upper_from_cells(layout, &cells);
        let gather_time = start.elapsed();
        let start = std::time::Instant::now();
        let hint_cols = Self::hint(&shape, params, &cells, &upper, &rows, threads);
        let hint_time = start.elapsed();
        let start = std::time::Instant::now();
        let offline = factored::offline_from_hint_columns_on(params, &hint_cols, threads);
        let packing_time = start.elapsed();
        (
            Self {
                shape,
                params,
                cells,
                records,
                upper,
                rows,
                hint_cols,
                offline,
                table_threads: 1,
            },
            PreprocessTimes {
                gather: gather_time,
                hint: hint_time,
                packing: packing_time,
            },
        )
    }

    /// Scalar reference for the batched gather and hint equality tests.
    #[cfg(test)]
    fn fill_column(
        shape: &PirShape,
        cells: &[u16],
        upper: &[Vec<u16>],
        rows: &[u32],
        col: usize,
        values: &mut [u16],
    ) {
        values.fill(0);
        for slot in 0..shape.slots {
            let (table, width) = if slot < shape.group_slots {
                (cells, shape.record_cells)
            } else {
                (upper[slot - shape.group_slots].as_slice(), UPPER_CELLS)
            };
            for pattern in 0..shape.patterns {
                let id = rows[pattern * shape.slots + slot];
                if id == NO_RECORD {
                    continue;
                }
                let source = &table[id as usize * width..][..width];
                let full = shape.slot_bits(slot) / CELL_BITS;
                let base = shape.slot_offset(slot);
                if col >= base && col < base + full {
                    values[pattern] = source[col - base];
                }
                for bit in 0..shape.slot_bits(slot) % CELL_BITS {
                    let dest = shape.full_columns() * CELL_BITS + shape.tail_bit_offset(slot) + bit;
                    if dest / CELL_BITS == col {
                        values[pattern] |= ((source[full] >> bit) & 1) << (dest % CELL_BITS);
                    }
                }
            }
        }
    }

    /// Gather each row tile once, then transpose it into adjacent plaintext
    /// columns. Full symbols have a single owner; only the packed suffix
    /// needs ORs. Missing records and padding are always explicitly zeroed.
    fn fill_columns(
        shape: &PirShape,
        cells: &[u16],
        upper: &[Vec<u16>],
        rows: HintRowIds<'_>,
        cols: std::ops::Range<usize>,
        values: &mut [u16],
    ) {
        assert!(!cols.is_empty() && cols.len() <= HINT_COLUMN_BATCH);
        assert!(cols.end <= shape.db_cols());
        assert_eq!(values.len() % cols.len(), 0);
        let stride = values.len() / cols.len();
        assert!(stride >= shape.patterns);
        let active = shape.row_cells().saturating_sub(cols.start).min(cols.len());
        for column in values[..active * stride].chunks_exact_mut(stride) {
            column[shape.patterns..].fill(0);
        }
        values[active * stride..].fill(0);
        if active == 0 {
            return;
        }

        let table_of = |slot: usize| {
            if slot < shape.group_slots {
                (cells, shape.record_cells)
            } else {
                (upper[slot - shape.group_slots].as_slice(), UPPER_CELLS)
            }
        };
        let full_sources = (0..shape.slots)
            .filter_map(|slot| {
                let packing = shape.packing(slot);
                let first = cols.start.max(packing.base);
                let end = cols.end.min(packing.base + packing.full);
                if first >= end {
                    return None;
                }
                let (table, width) = table_of(slot);
                Some((
                    slot * rows.slot_stride,
                    first - cols.start,
                    end - first,
                    first - packing.base,
                    table,
                    width,
                ))
            })
            .collect::<Vec<_>>();
        let tail_first = cols.start.max(shape.full_columns());
        let tail_end = cols.end.min(shape.row_cells());
        let tail_sources = (0..shape.slots)
            .filter_map(|slot| {
                let packing = shape.packing(slot);
                let first = tail_first.max(packing.tail_base);
                let end = tail_end.min(packing.tail_base + packing.tail_columns());
                if first >= end {
                    return None;
                }
                let (table, width) = table_of(slot);
                Some((slot * rows.slot_stride, packing, first, end, table, width))
            })
            .collect::<Vec<_>>();
        const TILE_ROWS: usize = 32;
        let tile_stride = active;
        let mut tile = vec![0u16; TILE_ROWS * tile_stride];
        for start in (0..shape.patterns).step_by(TILE_ROWS) {
            let row_count = TILE_ROWS.min(shape.patterns - start);
            for row in 0..row_count {
                let target = &mut tile[row * tile_stride..][..active];
                let pattern_base = (start + row) * rows.pattern_stride;
                for &(slot_base, column, count, offset, table, width) in &full_sources {
                    let out = &mut target[column..][..count];
                    let id = rows.values[slot_base + pattern_base];
                    if id == NO_RECORD {
                        out.fill(0);
                    } else {
                        out.copy_from_slice(&table[id as usize * width + offset..][..count]);
                    }
                }
                if tail_first < tail_end {
                    target[tail_first - cols.start..tail_end - cols.start].fill(0);
                    for &(slot_base, packing, first, end, table, width) in &tail_sources {
                        let id = rows.values[slot_base + pattern_base];
                        if id == NO_RECORD {
                            continue;
                        }
                        let value = ((table[id as usize * width + packing.full]
                            & ((1 << packing.tail_bits) - 1))
                            as u32)
                            << packing.shift;
                        for col in first..end {
                            target[col - cols.start] |= ((value
                                >> ((col - packing.tail_base) * CELL_BITS))
                                & ((1 << CELL_BITS) - 1))
                                as u16;
                        }
                    }
                }
            }
            transpose_hint_tile(
                &tile,
                tile_stride,
                row_count,
                values,
                stride,
                start,
                cfg!(target_arch = "x86_64"),
            );
        }
    }

    /// One range of hint columns, using the same bounded gather buffer for
    /// a complete build, a worker's partition, or a benchmark sample.
    fn hint_range(
        shape: &PirShape,
        params: &'static Params,
        cells: &[u16],
        upper: &[Vec<u16>],
        rows: HintRowIds<'_>,
        preprocessed_query: &[PolyMatrixNTT<'static>],
        range: std::ops::Range<usize>,
        batch_columns: usize,
    ) -> Vec<u64> {
        assert!((1..=HINT_COLUMN_BATCH).contains(&batch_columns));
        let db_rows = factored::simplepir_db_rows(params);
        let stride = db_rows + HINT_COLUMN_PADDING;
        let mut batch = vec![0u16; batch_columns.min(range.len()) * stride];
        let mut result = Vec::with_capacity(range.len() * params.poly_len);
        let mut kernel = factored::HintColumnKernel::new(params, preprocessed_query);
        for start in range.clone().step_by(batch_columns) {
            let end = (start + batch_columns).min(range.end);
            let values = &mut batch[..(end - start) * stride];
            Self::fill_columns(shape, cells, upper, rows, start..end, values);
            for column in values.chunks_exact(stride) {
                kernel.append_column(&column[..db_rows], &mut result);
            }
        }
        result
    }

    /// The SimplePIR hint, with adjacent columns gathered together from
    /// the stored records before invoking the unchanged YPIR hint kernel.
    fn hint(
        shape: &PirShape,
        params: &'static Params,
        cells: &[u16],
        upper: &[Vec<u16>],
        rows: &[u32],
        threads: usize,
    ) -> Vec<u64> {
        let db_cols = factored::simplepir_db_cols(params);
        let threads = threads.clamp(1, db_cols);
        let preprocessed_query = factored::pseudorandom_query(params, SEED_0);
        // Transpose only IDs, not stored records. This costs four bytes per
        // pattern slot, is shared by workers, and is included in hint time.
        let mut by_slot = vec![0u32; rows.len()];
        for slot in 0..shape.slots {
            for pattern in 0..shape.patterns {
                by_slot[slot * shape.patterns + pattern] = rows[pattern * shape.slots + slot];
            }
        }
        let row_ids = HintRowIds::by_slot(&by_slot, shape);
        let batch_columns =
            (HINT_GATHER_BYTES / threads / (shape.db_rows() * 2)).clamp(1, HINT_COLUMN_BATCH);
        let hint_range = |range| {
            Self::hint_range(
                shape,
                params,
                cells,
                upper,
                row_ids,
                &preprocessed_query,
                range,
                batch_columns,
            )
        };
        if threads == 1 {
            return hint_range(0..db_cols);
        }
        // The columns are independent: each thread takes a range of them.
        let per_thread = db_cols.div_ceil(threads);
        let ranges = (0..threads)
            .map(|k| (k * per_thread).min(db_cols)..((k + 1) * per_thread).min(db_cols))
            .filter(|range| !range.is_empty())
            .collect::<Vec<_>>();
        let hint_range = &hint_range;
        let parts = std::thread::scope(|scope| {
            let handles = ranges
                .iter()
                .cloned()
                .map(|range| scope.spawn(move || hint_range(range)))
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("hint worker"))
                .collect::<Vec<_>>()
        });
        // The ranges are in column order, so their parts concatenate.
        parts.concat()
    }

    /// Bring the database to the current fragment of the claim, for a log
    /// that kept its threshold and record size and did not shrink.
    ///
    /// The preprocessing is linear in the database, so only the difference
    /// is preprocessed. Records keep their numbers; full symbols keep their
    /// columns and the suffix moves intact when an upper slot is added.
    /// The difference is a list of runs: consecutive patterns
    /// whose record in one slot changed in the same way. Each run adds its
    /// change, times the hint of its row range, to the columns of its slot.
    /// The response-packing tables are then derived from the patched hint.
    /// With `changed`, only those records are read; it must list every
    /// record whose cells differ, as
    /// [`PrivateLayout::records_changed_by_append`] does after an append.
    pub fn update(
        &mut self,
        store: &UnauthStore,
        layout: &PrivateLayout,
        registry_index: usize,
        changed: Option<&[usize]>,
    ) -> UpdateTimes {
        self.update_records(layout, changed, &mut |record, cells| {
            entries_to_cells(
                &record_entries(store, layout, registry_index, record),
                cells,
            )
        })
    }

    /// [`Self::update`] from the primary-column entries of all segments of
    /// the longer log. With `changed`, only those records are read; it must
    /// list every record whose cells differ, as
    /// [`PrivateLayout::records_changed_by_append`] does after an append.
    pub fn update_from_entries(
        &mut self,
        layout: &PrivateLayout,
        entries: &[Fp],
        changed: Option<&[usize]>,
    ) -> UpdateTimes {
        let per_record = layout.entries_per_record();
        let mut padded = vec![crate::field::fp_zero(); per_record];
        self.update_records(layout, changed, &mut |record, cells| {
            let start = (record * per_record).min(entries.len());
            let end = ((record + 1) * per_record).min(entries.len());
            padded[..end - start].copy_from_slice(&entries[start..end]);
            padded[end - start..].fill(crate::field::fp_zero());
            entries_to_cells(&padded, cells);
        })
    }

    /// [`Self::update`] from the cells of the changed records read in
    /// advance: `cells` holds `record_cells` symbols for each record of
    /// `changed`, in order. This is the update a maintainer runs off its
    /// lock, on a clone of the database, while it keeps answering.
    pub fn update_from_cells(
        &mut self,
        layout: &PrivateLayout,
        changed: &[usize],
        cells: &[u16],
    ) -> UpdateTimes {
        let record_cells = self.shape.record_cells;
        assert_eq!(
            cells.len(),
            changed.len() * record_cells,
            "one record's cells per changed record"
        );
        let position: HashMap<usize, usize> = changed
            .iter()
            .enumerate()
            .map(|(k, &record)| (record, k))
            .collect();
        self.update_records(layout, Some(changed), &mut |record, out| {
            let k = position[&record];
            out.copy_from_slice(&cells[k * record_cells..(k + 1) * record_cells]);
        })
    }

    /// The cells of the records `changed` as stored for claim
    /// `registry_index`, in order, for [`Self::update_from_cells`].
    pub fn cells_of_records(
        store: &UnauthStore,
        layout: &PrivateLayout,
        registry_index: usize,
        changed: &[usize],
    ) -> Vec<u16> {
        let record_cells = PirShape::of(layout).record_cells;
        let mut cells = vec![0u16; changed.len() * record_cells];
        for (k, &record) in changed.iter().enumerate() {
            entries_to_cells(
                &record_entries(store, layout, registry_index, record),
                &mut cells[k * record_cells..(k + 1) * record_cells],
            );
        }
        cells
    }

    /// The update, with the cells of a record supplied by `cells_of`. When
    /// `changed` is absent every record is compared with its stored cells.
    fn update_records(
        &mut self,
        layout: &PrivateLayout,
        changed: Option<&[usize]>,
        cells_of: &mut dyn FnMut(usize, &mut [u16]),
    ) -> UpdateTimes {
        let start = std::time::Instant::now();
        let shape = PirShape::of(layout);
        assert_eq!(
            shape.record_bits, self.shape.record_bits,
            "record size changed"
        );
        assert_eq!(
            shape.record_cells, self.shape.record_cells,
            "record size changed"
        );
        assert!(shape.patterns >= self.shape.patterns, "the log shrank");
        assert!(shape.slots >= self.shape.slots, "the log shrank");
        let record_cells = shape.record_cells;
        let group_slots = shape.group_slots;
        assert_eq!(
            group_slots, self.shape.group_slots,
            "an update keeps the groups and the pattern size"
        );
        let records = layout.records();
        assert!(records >= self.records, "the log shrank");
        let params = shape.params();
        let db_cols = factored::simplepir_db_cols(params);

        // New cells, keeping the old cells of every record that changed.
        let mut old_cells: HashMap<usize, Vec<u16>> = HashMap::new();
        self.cells.resize(records * record_cells, 0);
        let mut scratch = vec![0u16; record_cells];
        let mut refresh = |record: usize, cells: &mut Vec<u16>| {
            cells_of(record, &mut scratch);
            let stored = &mut cells[record * record_cells..(record + 1) * record_cells];
            if stored != scratch.as_slice() {
                old_cells.insert(record, stored.to_vec());
                stored.copy_from_slice(&scratch);
            }
        };
        match changed {
            Some(changed) => {
                for &record in changed {
                    assert!(record < records, "changed record outside the layout");
                    refresh(record, &mut self.cells);
                }
            }
            None => {
                for record in 0..records {
                    refresh(record, &mut self.cells);
                }
            }
        }

        // New rows and new upper nodes.
        let rows = layout.rows_table();
        let old_upper = std::mem::take(&mut self.upper);
        let upper = Self::upper_from_cells(layout, &self.cells);

        // Extend the hint before moving its suffix to make space for any
        // newly appended upper slots.
        let old_cols = factored::simplepir_db_cols(self.params);
        let poly_len = params.poly_len;
        assert!(db_cols >= old_cols, "the hint lost columns");
        self.hint_cols.resize(db_cols * poly_len, 0);
        // New upper slots insert full symbols immediately before the
        // suffix. Move the existing suffix intact, including its partial
        // last symbol, before applying linear deltas for changed/new slots.
        let old_full = self.shape.full_columns();
        let new_full = shape.full_columns();
        if old_full != new_full {
            self.hint_cols.copy_within(
                old_full * poly_len..self.shape.row_cells() * poly_len,
                new_full * poly_len,
            );
            self.hint_cols[old_full * poly_len..new_full * poly_len].fill(0);
        }

        let preprocessed_query = factored::pseudorandom_query(params, SEED_0);
        let old_shape = self.shape;
        let old_rows = &self.rows;
        let old_record_at = |pattern: usize, slot: usize| -> u32 {
            if pattern < old_shape.patterns && slot < old_shape.slots {
                old_rows[pattern * old_shape.slots + slot]
            } else {
                NO_RECORD
            }
        };
        fn upper_cells_of(table: &[Vec<u16>], level: usize, pos: usize) -> Option<&[u16]> {
            let cells = table.get(level)?;
            cells.get(pos * UPPER_CELLS..(pos + 1) * UPPER_CELLS)
        }
        let changed = |slot: usize, old: u32, new: u32| -> bool {
            if old != new || new == NO_RECORD {
                return old != new;
            }
            let id = new as usize;
            if slot < group_slots {
                old_cells.contains_key(&id)
            } else {
                upper_cells_of(&old_upper, slot - group_slots, id)
                    != upper_cells_of(&upper, slot - group_slots, id)
            }
        };
        // The change of one slot is a sum over its runs: every column of the
        // slot gains its cell's change times the hint of the run's row
        // range. A column of the hint is contiguous, so the sum of one
        // column over the runs is a scaled vector add per run, in the two
        // residues of the ciphertext modulus. The change of a cell is a
        // small signed number and a residue is below 2^28, so the signed
        // sums of this many products stay within the bound of
        // [`Finalize`].
        const REDUCE_EVERY: usize = 1 << 13;
        let finalize = Finalize::of(params);
        let mut sum_lo = vec![0i64; poly_len];
        let mut sum_hi = vec![0i64; poly_len];
        let hint_cols = &mut self.hint_cols;
        for slot in 0..shape.slots {
            let packing = shape.packing(slot);
            let width = packing.columns();
            let mut runs: Vec<(Option<&[u16]>, Option<&[u16]>, std::ops::Range<usize>)> =
                Vec::new();
            let mut pattern = 0usize;
            while pattern < shape.patterns {
                let old = old_record_at(pattern, slot);
                let new = rows[pattern * shape.slots + slot];
                if !changed(slot, old, new) {
                    pattern += 1;
                    continue;
                }
                let start = pattern;
                while pattern < shape.patterns
                    && old_record_at(pattern, slot) == old
                    && rows[pattern * shape.slots + slot] == new
                {
                    pattern += 1;
                }
                let before: Option<&[u16]> = (old != NO_RECORD).then(|| {
                    let id = old as usize;
                    if slot < group_slots {
                        match old_cells.get(&id) {
                            Some(cells) => cells.as_slice(),
                            None => &self.cells[id * record_cells..(id + 1) * record_cells],
                        }
                    } else {
                        upper_cells_of(&old_upper, slot - group_slots, id).expect("old upper node")
                    }
                });
                let after: Option<&[u16]> = (new != NO_RECORD).then(|| {
                    let id = new as usize;
                    if slot < group_slots {
                        &self.cells[id * record_cells..(id + 1) * record_cells]
                    } else {
                        upper_cells_of(&upper, slot - group_slots, id).expect("new upper node")
                    }
                });
                runs.push((before, after, start..pattern));
            }
            if runs.is_empty() {
                continue;
            }
            // The hint of each run's row range, in the two residues.
            let range_hints = runs
                .iter()
                .map(|(_, _, row_range)| {
                    let values =
                        factored::hint_of_row_range(params, &preprocessed_query, row_range.clone());
                    (
                        values
                            .iter()
                            .map(|&v| v % params.moduli[0])
                            .collect::<Vec<u64>>(),
                        values
                            .iter()
                            .map(|&v| v % params.moduli[1])
                            .collect::<Vec<u64>>(),
                    )
                })
                .collect::<Vec<_>>();
            for offset in 0..width {
                let mut touched = false;
                let mut pending = 0usize;
                for ((before, after, _), (range_lo, range_hi)) in runs.iter().zip(&range_hints) {
                    let delta = after.map_or(0, |cells| packing.cell(cells, offset) as i64)
                        - before.map_or(0, |cells| packing.cell(cells, offset) as i64);
                    if delta == 0 {
                        continue;
                    }
                    if !touched {
                        sum_lo.fill(0);
                        sum_hi.fill(0);
                        touched = true;
                    }
                    add_scaled_residues(&mut sum_lo, &mut sum_hi, delta, range_lo, range_hi);
                    pending += 1;
                    if pending == REDUCE_EVERY {
                        finalize.reduce_sums(&mut sum_lo, &mut sum_hi);
                        pending = 0;
                    }
                }
                if !touched {
                    continue;
                }
                let col = packing.column(offset);
                let column = &mut hint_cols[col * poly_len..(col + 1) * poly_len];
                finalize.add_sums_to_column(&sum_lo, &sum_hi, column);
            }
        }

        let hint_time = start.elapsed();

        // The tables of the old hint are released before the new ones are
        // derived, so one set is held at a time.
        let start = std::time::Instant::now();
        self.offline.prepacked_lwe = Vec::new();
        self.offline.precomp = Vec::new();
        self.offline =
            factored::offline_from_hint_columns_on(params, &self.hint_cols, self.table_threads);
        let packing_time = start.elapsed();
        self.shape = shape;
        self.params = params;
        self.records = records;
        self.upper = upper;
        self.rows = rows;
        UpdateTimes {
            hint: hint_time,
            packing: packing_time,
        }
    }

    /// The hint of the listed database columns, computed from the stored
    /// records: `poly_len` values per column. It equals the stored hint of
    /// those columns, preserving the requested order and duplicates.
    /// Adjacent requests share the full builder's gather batches; timing
    /// isolated columns does not capture that memory-access saving.
    pub fn hint_of_columns(&self, cols: &[usize]) -> Vec<Vec<u64>> {
        assert!(cols.iter().all(|&col| col < self.shape.db_cols()));
        if cols.is_empty() {
            return Vec::new();
        }
        let preprocessed_query = factored::pseudorandom_query(self.params, SEED_0);
        let mut result = Vec::with_capacity(cols.len());
        let mut start = 0;
        while start < cols.len() {
            let mut end = start + 1;
            while end < cols.len() && cols[end] == cols[end - 1] + 1 {
                end += 1;
            }
            let hint = Self::hint_range(
                &self.shape,
                self.params,
                &self.cells,
                &self.upper,
                HintRowIds::by_pattern(&self.rows, &self.shape),
                &preprocessed_query,
                cols[start]..cols[end - 1] + 1,
                (HINT_GATHER_BYTES / (self.shape.db_rows() * 2)).clamp(1, HINT_COLUMN_BATCH),
            );
            result.extend(
                hint.chunks_exact(self.params.poly_len)
                    .map(|column| column.to_vec()),
            );
            start = end;
        }
        result
    }

    /// Derive the response-packing table of one response ciphertext from
    /// the stored hint and return the bytes that the state keeps of it.
    /// The tables of a database are those of its [`PirShape::outputs`]
    /// ciphertexts, each derived this way; the prepacked columns of the
    /// output are an input to the table and are released with it.
    pub fn packing_table_of_output(&self, output: usize) -> usize {
        let (_prepacked, (first, rest, _)) = factored::packing_table_of_columns(
            self.params,
            &self.hint_cols,
            output,
            &self.offline.fake_pack_pub_params,
            &self.offline.y_constants,
        );
        first.as_slice().len() * 8
            + rest
                .iter()
                .map(|table| table.as_slice().len() * 8)
                .sum::<usize>()
    }

    /// The stored hint of one database column: `poly_len` values.
    pub fn stored_hint_column(&self, col: usize) -> Vec<u64> {
        let n = self.params.poly_len;
        self.hint_cols[col * n..(col + 1) * n].to_vec()
    }

    /// Bytes of the preprocessing: the hint and the response-packing tables.
    pub fn preprocessed_bytes(&self) -> usize {
        self.hint_bytes() + self.packing_bytes()
    }

    /// Bytes of the hint: `poly_len` words per database column.
    pub fn hint_bytes(&self) -> usize {
        self.hint_cols.len() * 8
    }

    /// Bytes of the response-packing tables.
    pub fn packing_bytes(&self) -> usize {
        packing_table_bytes(&self.offline)
    }

    /// The blocks of `poly_len` rows of one database column that are all
    /// zero, out of all such blocks: the hint's kernel does no arithmetic
    /// for an all-zero block. The columns are gathered as the hint gathers
    /// them; this is a separate pass, for reports.
    pub fn zero_hint_blocks(&self) -> (u64, u64) {
        let shape = &self.shape;
        let db_rows = factored::simplepir_db_rows(self.params);
        let db_cols = shape.db_cols();
        let stride = db_rows + HINT_COLUMN_PADDING;
        let batch = HINT_COLUMN_BATCH.min(db_cols);
        let mut values = vec![0u16; batch * stride];
        let rows = HintRowIds::by_pattern(&self.rows, shape);
        let mut zero = 0u64;
        for start in (0..db_cols).step_by(batch) {
            let end = (start + batch).min(db_cols);
            let values = &mut values[..(end - start) * stride];
            Self::fill_columns(shape, &self.cells, &self.upper, rows, start..end, values);
            for column in values.chunks_exact(stride) {
                zero += count_zero_blocks(&column[..db_rows], self.params.poly_len);
            }
        }
        (zero, (db_cols * (db_rows / self.params.poly_len)) as u64)
    }

    /// Bytes of the stored records and of the table that places them in
    /// rows.
    pub fn record_state_bytes(&self) -> usize {
        let upper = self.upper.iter().map(Vec::len).sum::<usize>();
        (self.cells.len() + upper) * 2 + self.rows.len() * 4
    }

    /// The plaintext cells of the upper node `level` levels above the
    /// groups at position `pos`.
    pub fn upper_cells(&self, level: usize, pos: usize) -> &[u16] {
        &self.upper[level][pos * UPPER_CELLS..(pos + 1) * UPPER_CELLS]
    }

    /// The plaintext cells of the item in one slot of a record: a group in
    /// a group slot, an upper node above.
    pub fn slot_cells(&self, slot: usize, id: usize) -> &[u16] {
        let group_slots = self.shape.group_slots;
        if slot < group_slots {
            self.record_cells(id)
        } else {
            self.upper_cells(slot - group_slots, id)
        }
    }

    /// Dimensions of this database.
    pub fn shape(&self) -> PirShape {
        self.shape
    }

    /// The cells of every record, record by record.
    pub fn cells(&self) -> &[u16] {
        &self.cells
    }

    /// Release the preprocessing and return the owned plaintext cells.
    /// A fresh build can reuse this buffer without retaining two databases.
    pub fn into_cells(self) -> Vec<u16> {
        self.cells
    }

    /// The plaintext cells of one record.
    pub fn record_cells(&self, record: usize) -> &[u16] {
        &self.cells[record * self.shape.record_cells..(record + 1) * self.shape.record_cells]
    }

    /// The record in each slot of a row.
    pub fn row(&self, pattern: usize) -> Vec<Option<usize>> {
        self.rows[pattern * self.shape.slots..(pattern + 1) * self.shape.slots]
            .iter()
            .map(|&record| (record != NO_RECORD).then_some(record as usize))
            .collect()
    }

    /// The SimplePIR hint of the database, row-major as \textsc{YPIR} lays
    /// it out: `[i * db_cols + col]`.
    pub fn hint_values(&self) -> Vec<u64> {
        let n = self.params.poly_len;
        let db_cols = factored::simplepir_db_cols(self.params);
        let mut hint = vec![0u64; db_cols * n];
        for col in 0..db_cols {
            for i in 0..n {
                hint[i * db_cols + col] = self.hint_cols[col * n + i];
            }
        }
        hint
    }

    /// The first pass of a query: its inner product with every database
    /// column, computed from the stored records.
    ///
    /// For each slot, the coefficients of the patterns that place a record
    /// there are added, and each sum is multiplied with the record once.
    pub fn first_pass(&self, query: &[u64]) -> Result<Vec<u64>, PirError> {
        let params = self.params;
        let shape = &self.shape;
        if query.len() != factored::simplepir_db_rows(params) {
            return Err(PirError::Query);
        }
        let mut out = vec![0u64; factored::simplepir_db_cols(params)];
        let mut acc_lo = vec![0u64; self.records];
        let mut acc_hi = vec![0u64; self.records];
        let group_cells = shape.group_slots * shape.record_cells;
        let mut sum_lo = vec![0u64; group_cells];
        let mut sum_hi = vec![0u64; group_cells];
        let packing = (0..shape.slots)
            .map(|slot| shape.packing(slot))
            .collect::<Vec<_>>();
        let mut high_lo = vec![0u64; shape.slots];
        let mut high_hi = vec![0u64; shape.slots];
        // Keep the separately stored cells during accumulation. Pack their
        // residual-bit sums only when writing the final PIR columns.
        let tail_mask = if shape.record_bits % CELL_BITS == 0 {
            u16::MAX
        } else {
            (1u16 << (shape.record_bits % CELL_BITS)) - 1
        };
        // A reduced coefficient is below 2^32 and a cell below 2^16, so this
        // many products fit in 64 bits.
        const REDUCE_EVERY: usize = 1 << 15;

        // Group `g` sits in slot `g % group_slots`, so one pass over the rows
        // sums the coefficients of every group.
        for (pattern, row) in self.rows.chunks_exact(shape.slots).enumerate() {
            let coefficient_lo = query[pattern] & LOW_MASK;
            let coefficient_hi = query[pattern] >> 32;
            for (slot, &record) in row[..shape.group_slots].iter().enumerate() {
                if record == NO_RECORD {
                    continue;
                }
                let g = record as usize;
                debug_assert_eq!(g % shape.group_slots, slot);
                // Coefficients are below 2^32; reduce before the sum can
                // reach 2^64.
                acc_lo[g] += coefficient_lo;
                acc_hi[g] += coefficient_hi;
                if acc_lo[g] >= 1 << 62 {
                    acc_lo[g] = barrett_coeff_u64(params, acc_lo[g], 0);
                }
                if acc_hi[g] >= 1 << 62 {
                    acc_hi[g] = barrett_coeff_u64(params, acc_hi[g], 1);
                }
            }
        }
        // Four adjacent slot cycles form a tile. Within it, multiply four
        // groups assigned to the same slot together: each SIMD accumulator
        // is loaded and stored once per batch, instead of once per group.
        // The stored layout is unchanged and every group is visited once.
        const B: usize = 4;
        let mut pending = vec![0usize; shape.group_slots];
        for block in (0..self.records).step_by(B * shape.group_slots) {
            for slot in 0..shape.group_slots {
                let first = block + slot;
                if first >= self.records {
                    break;
                }
                let count = B.min((self.records - 1 - first) / shape.group_slots + 1);
                let base = slot * shape.record_cells;
                let slot_lo = &mut sum_lo[base..base + shape.record_cells];
                let slot_hi = &mut sum_hi[base..base + shape.record_cells];
                if pending[slot] + count > REDUCE_EVERY {
                    for (lo, hi) in slot_lo.iter_mut().zip(slot_hi.iter_mut()) {
                        *lo = barrett_coeff_u64(params, *lo, 0);
                        *hi = barrett_coeff_u64(params, *hi, 1);
                    }
                    high_lo[slot] = barrett_coeff_u64(params, high_lo[slot], 0);
                    high_hi[slot] = barrett_coeff_u64(params, high_hi[slot], 1);
                    pending[slot] = 0;
                }
                let mut coeff_lo = [0u32; B];
                let mut coeff_hi = [0u32; B];
                let mut records = [self.record_cells(first); B];
                let p = packing[slot];
                for r in 0..count {
                    let g = first + r * shape.group_slots;
                    let a_lo = barrett_coeff_u64(params, acc_lo[g], 0);
                    let a_hi = barrett_coeff_u64(params, acc_hi[g], 1);
                    coeff_lo[r] = a_lo as u32;
                    coeff_hi[r] = a_hi as u32;
                    records[r] = self.record_cells(g);
                }
                if coeff_lo.iter().chain(&coeff_hi).any(|&a| a != 0) {
                    add_scaled_batch(slot_lo, slot_hi, &coeff_lo, &coeff_hi, &records, tail_mask);
                    // Bit extraction cannot be deferred across addition.
                    // Only a tail crossing a symbol boundary needs this
                    // extra sum; all other tails are shifted after the scan.
                    if p.tail_columns() == 2 {
                        for r in 0..count {
                            let high =
                                ((records[r][p.full] & tail_mask) >> (CELL_BITS - p.shift)) as u64;
                            high_lo[slot] += coeff_lo[r] as u64 * high;
                            high_hi[slot] += coeff_hi[r] as u64 * high;
                        }
                    }
                }
                pending[slot] += count;
            }
        }
        for slot in 0..shape.group_slots {
            let base = slot * shape.record_cells;
            finalize_slot(
                params,
                &mut out,
                &sum_lo[base..base + shape.record_cells],
                &sum_hi[base..base + shape.record_cells],
                high_lo[slot],
                high_hi[slot],
                packing[slot],
            );
        }

        // Above the groups, the coefficient of a node is the sum of the
        // query over the patterns below its sibling: at level `j`, entry
        // `k` of `subtree_query` is the sum over `[k 2^j, (k+1) 2^j)`, and
        // node `pos` uses entry `pos ^ 1`. The entries are folded pairwise
        // from one level to the next, so every coefficient is read once.
        let mut subtree_query = if shape.upper_slots() == 0 {
            Vec::new()
        } else {
            query[..shape.patterns]
                .iter()
                .map(|&coefficient| {
                    let lo = barrett_coeff_u64(params, coefficient & LOW_MASK, 0);
                    let hi = barrett_coeff_u64(params, coefficient >> 32, 1);
                    lo | (hi << 32)
                })
                .collect::<Vec<_>>()
        };
        let mut upper_lo = [0u64; UPPER_CELLS];
        let mut upper_hi = [0u64; UPPER_CELLS];
        for level in 0..shape.upper_slots() {
            upper_lo.fill(0);
            upper_hi.fill(0);
            let mut pending = 0usize;
            let count = self.upper[level].len() / UPPER_CELLS;
            debug_assert_eq!(subtree_query.len(), count);
            for pos in 0..count {
                // An odd final subtree has no stored right sibling.
                let coefficient = subtree_query.get(pos ^ 1).copied().unwrap_or(0);
                let a_lo = coefficient & LOW_MASK;
                let a_hi = coefficient >> 32;
                if a_lo == 0 && a_hi == 0 {
                    continue;
                }
                let p = packing[shape.group_slots + level];
                add_scaled_batch(
                    &mut upper_lo,
                    &mut upper_hi,
                    &[a_lo as u32],
                    &[a_hi as u32],
                    &[self.upper_cells(level, pos)],
                    (1 << p.tail_bits) - 1,
                );
                if p.shift + p.tail_bits > CELL_BITS {
                    let high = ((self.upper_cells(level, pos)[p.full] & ((1 << p.tail_bits) - 1))
                        >> (CELL_BITS - p.shift)) as u64;
                    high_lo[shape.group_slots + level] += a_lo * high;
                    high_hi[shape.group_slots + level] += a_hi * high;
                }
                pending += 1;
                if pending == REDUCE_EVERY {
                    for (lo, hi) in upper_lo.iter_mut().zip(upper_hi.iter_mut()) {
                        *lo = barrett_coeff_u64(params, *lo, 0);
                        *hi = barrett_coeff_u64(params, *hi, 1);
                    }
                    high_lo[shape.group_slots + level] =
                        barrett_coeff_u64(params, high_lo[shape.group_slots + level], 0);
                    high_hi[shape.group_slots + level] =
                        barrett_coeff_u64(params, high_hi[shape.group_slots + level], 1);
                    pending = 0;
                }
            }
            let slot = shape.group_slots + level;
            finalize_slot(
                params,
                &mut out,
                &upper_lo,
                &upper_hi,
                high_lo[slot],
                high_hi[slot],
                packing[slot],
            );
            if level + 1 < shape.upper_slots() {
                fold_query_pairs(params, &mut subtree_query);
            }
        }
        Ok(out)
    }

    /// Answer one query: the first pass over the stored records, then YPIR's
    /// response packing.
    pub fn answer(&self, query: &[u64], pack_keys: &PackKeys) -> Result<Vec<Vec<u8>>, PirError> {
        let pack_keys = pack_keys.to_matrices(self.params)?;
        let first_pass = self.first_pass(query)?;
        Ok(factored::response_from_first_pass(
            self.params,
            &self.offline,
            &first_pass,
            &pack_keys,
        ))
    }
}

/// Write a slot's accumulated symbols into the packed PIR row. If a tail
/// crosses a symbol boundary, `high_*` accumulated its high bits before
/// summation. Recover its low-part sum as S - 2^cut H modulo each prime;
/// shifting or masking the sum itself would not implement that split.
fn finalize_slot(
    params: &Params,
    out: &mut [u64],
    sum_lo: &[u64],
    sum_hi: &[u64],
    high_lo: u64,
    high_hi: u64,
    packing: SlotPacking,
) {
    for i in 0..packing.full {
        let lo = barrett_coeff_u64(params, sum_lo[i], 0);
        let hi = barrett_coeff_u64(params, sum_hi[i], 1);
        out[packing.base + i] = barrett_u64(params, params.crt_compose_2(lo, hi));
    }
    if packing.tail_bits == 0 {
        return;
    }
    let mut lo = barrett_coeff_u64(params, sum_lo[packing.full], 0);
    let mut hi = barrett_coeff_u64(params, sum_hi[packing.full], 1);
    if packing.tail_columns() == 2 {
        let high_lo = barrett_coeff_u64(params, high_lo, 0);
        let high_hi = barrett_coeff_u64(params, high_hi, 1);
        let cut = CELL_BITS - packing.shift;
        lo = barrett_coeff_u64(
            params,
            lo + params.moduli[0] - barrett_coeff_u64(params, high_lo << cut, 0),
            0,
        );
        hi = barrett_coeff_u64(
            params,
            hi + params.moduli[1] - barrett_coeff_u64(params, high_hi << cut, 1),
            1,
        );
        add_packed_sum(params, &mut out[packing.tail_base + 1], high_lo, high_hi);
    }
    lo = barrett_coeff_u64(params, lo << packing.shift, 0);
    hi = barrett_coeff_u64(params, hi << packing.shift, 1);
    add_packed_sum(params, &mut out[packing.tail_base], lo, hi);
}

/// Several slots contribute disjoint fields to one suffix column. Their
/// dot products add modulo Q; both summands are below Q and 2Q fits u64.
fn add_packed_sum(params: &Params, out: &mut u64, lo: u64, hi: u64) {
    *out += barrett_u64(params, params.crt_compose_2(lo, hi));
    if *out >= params.modulus {
        *out -= params.modulus;
    }
}

/// Replace canonical packed CRT coefficients by their pairwise sums.
/// Each residue is below its modulus, so a sum is below twice the modulus
/// and one subtraction suffices. For an odd length the last pair adds zero.
fn fold_query_pairs(params: &Params, query: &mut Vec<u64>) {
    let pairs = query.len().div_ceil(2);
    for pair in 0..pairs {
        let left = query[2 * pair];
        let right = query.get(2 * pair + 1).copied().unwrap_or(0);
        let mut lo = (left & LOW_MASK) + (right & LOW_MASK);
        let mut hi = (left >> 32) + (right >> 32);
        if lo >= params.moduli[0] {
            lo -= params.moduli[0];
        }
        if hi >= params.moduli[1] {
            hi -= params.moduli[1];
        }
        query[pair] = lo | (hi << 32);
    }
    query.truncate(pairs);
}

/// Accumulate a small batch of equal-width records assigned to one slot.
/// Coefficients are reduced below 2^32. `mask` discards unused bits of the
/// last stored cell, while full cells retain the whole u16 arithmetic range.
fn add_scaled_batch<const B: usize>(
    sum_lo: &mut [u64],
    sum_hi: &mut [u64],
    coeff_lo: &[u32; B],
    coeff_hi: &[u32; B],
    records: &[&[u16]; B],
    mask: u16,
) {
    assert!(B > 0);
    assert_eq!(sum_lo.len(), sum_hi.len());
    assert!(records.iter().all(|cells| cells.len() == sum_lo.len()));
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 is available and all slices have the same length.
        unsafe { add_scaled_batch_avx2(sum_lo, sum_hi, coeff_lo, coeff_hi, records, mask) };
        return;
    }
    add_scaled_batch_scalar(sum_lo, sum_hi, coeff_lo, coeff_hi, records, mask);
}

fn add_scaled_batch_scalar<const B: usize>(
    sum_lo: &mut [u64],
    sum_hi: &mut [u64],
    coeff_lo: &[u32; B],
    coeff_hi: &[u32; B],
    records: &[&[u16]; B],
    mask: u16,
) {
    let n = sum_lo.len();
    for r in 0..B {
        for k in 0..n {
            let cell = (records[r][k] & if k + 1 == n { mask } else { u16::MAX }) as u64;
            sum_lo[k] += coeff_lo[r] as u64 * cell;
            sum_hi[k] += coeff_hi[r] as u64 * cell;
        }
    }
}

/// Whether this host runs the AVX2 kernels.
#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    use std::sync::OnceLock;
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| is_x86_feature_detected!("avx2"))
}

/// The arithmetic that turns the signed sums of a patched column into its
/// new hint values: reduction modulo each prime, Garner's composition, and
/// the addition modulo the ciphertext modulus, in integer operations whose
/// operands stay below `2^32` so that the AVX2 kernel computes them
/// exactly.
///
/// A sum is a signed number of magnitude below `2^57`; `bias = q << 30` is
/// a multiple of the prime that makes it non-negative and below `2^59`.
/// For `x < 2^59` and `2^27 < q < 2^28`, `t = ((x >> 27) * m) >> 32` with
/// `m = floor(2^59 / q)` is at most `x / q` and above `x / q - 3`, so
/// `x - t * q` is below `3q` and two conditional subtractions reduce it.
#[derive(Clone, Copy)]
struct Finalize {
    q0: u64,
    q1: u64,
    m0: u64,
    m1: u64,
    bias0: i64,
    bias1: i64,
    /// `q0^-1 mod q1`, the constant of Garner's composition
    /// `lo + q0 * ((hi - lo) * inverse mod q1)`.
    inverse: u64,
    modulus: u64,
}

impl Finalize {
    fn of(params: &Params) -> Self {
        let (q0, q1) = (params.moduli[0], params.moduli[1]);
        for q in [q0, q1] {
            assert!(q > 1 << 27 && q < 1 << 28, "a prime of 28 bits");
        }
        assert!(
            q0 < 2 * q1,
            "a residue of q0 reduces modulo q1 by one subtraction"
        );
        // `mod0_inv_mod1` is `q0 * (q0^-1 mod q1)`.
        let inverse = params.mod0_inv_mod1 / q0;
        assert_eq!((inverse * q0) % q1, 1, "the inverse of q0 modulo q1");
        Self {
            q0,
            q1,
            m0: (1u64 << 59) / q0,
            m1: (1u64 << 59) / q1,
            bias0: (q0 << 30) as i64,
            bias1: (q1 << 30) as i64,
            inverse,
            modulus: params.modulus,
        }
    }

    /// `x mod q` for `x < 2^59`.
    #[inline]
    fn reduce(x: u64, q: u64, m: u64) -> u64 {
        let t = ((x >> 27) * m) >> 32;
        let mut r = x - t * q;
        if r >= 2 * q {
            r -= 2 * q;
        }
        if r >= q {
            r -= q;
        }
        r
    }

    /// The value below the ciphertext modulus with the given residues.
    #[inline]
    fn compose(&self, lo: u64, hi: u64) -> u64 {
        let lo1 = if lo >= self.q1 { lo - self.q1 } else { lo };
        let diff = if hi >= lo1 {
            hi - lo1
        } else {
            hi + self.q1 - lo1
        };
        lo + self.q0 * Self::reduce(diff * self.inverse, self.q1, self.m1)
    }

    /// Reduce the sums in place to the residues they stand for.
    fn reduce_sums(&self, sum_lo: &mut [i64], sum_hi: &mut [i64]) {
        for (lo, hi) in sum_lo.iter_mut().zip(sum_hi.iter_mut()) {
            *lo = Self::reduce((*lo + self.bias0) as u64, self.q0, self.m0) as i64;
            *hi = Self::reduce((*hi + self.bias1) as u64, self.q1, self.m1) as i64;
        }
    }

    /// `column[k] = (column[k] + value) mod modulus`, where `value` is the
    /// number below the modulus with the residues of `sum_lo[k]` and
    /// `sum_hi[k]`.
    fn add_sums_to_column(&self, sum_lo: &[i64], sum_hi: &[i64], column: &mut [u64]) {
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: the host supports AVX2.
            unsafe { self.add_sums_to_column_avx2(sum_lo, sum_hi, column) };
            return;
        }
        self.add_sums_to_column_scalar(sum_lo, sum_hi, column);
    }

    fn add_sums_to_column_scalar(&self, sum_lo: &[i64], sum_hi: &[i64], column: &mut [u64]) {
        for ((value, &lo), &hi) in column.iter_mut().zip(sum_lo).zip(sum_hi) {
            let lo = Self::reduce((lo + self.bias0) as u64, self.q0, self.m0);
            let hi = Self::reduce((hi + self.bias1) as u64, self.q1, self.m1);
            *value += self.compose(lo, hi);
            if *value >= self.modulus {
                *value -= self.modulus;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_sums_to_column_avx2(&self, sum_lo: &[i64], sum_hi: &[i64], column: &mut [u64]) {
        use std::arch::x86_64::*;
        let n = column.len();
        let mut k = 0;
        // SAFETY: every access stays within the slices, four lanes at a time.
        unsafe {
            let splat = |v: u64| _mm256_set1_epi64x(v as i64);
            let (q0, q1) = (splat(self.q0), splat(self.q1));
            let (m0, m1) = (splat(self.m0), splat(self.m1));
            let (bias0, bias1) = (splat(self.bias0 as u64), splat(self.bias1 as u64));
            let (q0_less, q1_less) = (splat(self.q0 - 1), splat(self.q1 - 1));
            let (q0_twice, q1_twice) = (splat(2 * self.q0), splat(2 * self.q1));
            let (q0_twice_less, q1_twice_less) = (splat(2 * self.q0 - 1), splat(2 * self.q1 - 1));
            let inverse = splat(self.inverse);
            let modulus = splat(self.modulus);
            let modulus_less = splat(self.modulus - 1);
            let zero = _mm256_setzero_si256();
            // `x mod q` for lanes `x < 2^59`, as in [`Finalize::reduce`].
            let reduce = |x: __m256i, q, q_less, q_twice, q_twice_less, m| {
                let t = _mm256_srli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(x, 27), m), 32);
                let r = _mm256_sub_epi64(x, _mm256_mul_epu32(t, q));
                let r = _mm256_sub_epi64(
                    r,
                    _mm256_and_si256(_mm256_cmpgt_epi64(r, q_twice_less), q_twice),
                );
                _mm256_sub_epi64(r, _mm256_and_si256(_mm256_cmpgt_epi64(r, q_less), q))
            };
            while k + 4 <= n {
                let lo = _mm256_add_epi64(
                    _mm256_loadu_si256(sum_lo.as_ptr().add(k) as *const _),
                    bias0,
                );
                let hi = _mm256_add_epi64(
                    _mm256_loadu_si256(sum_hi.as_ptr().add(k) as *const _),
                    bias1,
                );
                let lo = reduce(lo, q0, q0_less, q0_twice, q0_twice_less, m0);
                let hi = reduce(hi, q1, q1_less, q1_twice, q1_twice_less, m1);
                // Garner: lo + q0 * ((hi - lo) * inverse mod q1).
                let lo1 =
                    _mm256_sub_epi64(lo, _mm256_and_si256(_mm256_cmpgt_epi64(lo, q1_less), q1));
                let diff = _mm256_sub_epi64(hi, lo1);
                let diff =
                    _mm256_add_epi64(diff, _mm256_and_si256(_mm256_cmpgt_epi64(zero, diff), q1));
                let t = reduce(
                    _mm256_mul_epu32(diff, inverse),
                    q1,
                    q1_less,
                    q1_twice,
                    q1_twice_less,
                    m1,
                );
                let add = _mm256_add_epi64(lo, _mm256_mul_epu32(q0, t));
                let ptr = column.as_mut_ptr().add(k) as *mut __m256i;
                let value = _mm256_add_epi64(_mm256_loadu_si256(ptr), add);
                let value = _mm256_sub_epi64(
                    value,
                    _mm256_and_si256(_mm256_cmpgt_epi64(value, modulus_less), modulus),
                );
                _mm256_storeu_si256(ptr, value);
                k += 4;
            }
        }
        self.add_sums_to_column_scalar(&sum_lo[k..], &sum_hi[k..], &mut column[k..]);
    }
}

/// `sum_lo[k] += delta * src_lo[k]` and `sum_hi[k] += delta * src_hi[k]`:
/// the product of one run's change with the hint of its rows, in the two
/// residues. The change is a signed difference of two symbols, so it fits
/// a 32-bit lane, and a residue is below `2^28`.
fn add_scaled_residues(
    sum_lo: &mut [i64],
    sum_hi: &mut [i64],
    delta: i64,
    src_lo: &[u64],
    src_hi: &[u64],
) {
    debug_assert!(delta.unsigned_abs() < 1 << 16, "a difference of symbols");
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: the host supports AVX2.
        unsafe { add_scaled_residues_avx2(sum_lo, sum_hi, delta, src_lo, src_hi) };
        return;
    }
    for (((lo, hi), &x), &y) in sum_lo
        .iter_mut()
        .zip(sum_hi.iter_mut())
        .zip(src_lo)
        .zip(src_hi)
    {
        *lo += delta * x as i64;
        *hi += delta * y as i64;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_residues_avx2(
    sum_lo: &mut [i64],
    sum_hi: &mut [i64],
    delta: i64,
    src_lo: &[u64],
    src_hi: &[u64],
) {
    use std::arch::x86_64::*;
    let n = src_lo.len();
    let lo_ptr = sum_lo.as_mut_ptr();
    let hi_ptr = sum_hi.as_mut_ptr();
    let x_ptr = src_lo.as_ptr();
    let y_ptr = src_hi.as_ptr();
    let mut k = 0;
    // SAFETY: every access stays within the slices, four lanes at a time.
    // The multiply takes the low 32 bits of each lane as signed numbers:
    // the change, and a residue below 2^28.
    unsafe {
        let delta_v = _mm256_set1_epi64x(delta);
        while k + 4 <= n {
            let x = _mm256_loadu_si256(x_ptr.add(k) as *const _);
            let y = _mm256_loadu_si256(y_ptr.add(k) as *const _);
            let lo = _mm256_loadu_si256(lo_ptr.add(k) as *const _);
            let hi = _mm256_loadu_si256(hi_ptr.add(k) as *const _);
            _mm256_storeu_si256(
                lo_ptr.add(k) as *mut _,
                _mm256_add_epi64(lo, _mm256_mul_epi32(delta_v, x)),
            );
            _mm256_storeu_si256(
                hi_ptr.add(k) as *mut _,
                _mm256_add_epi64(hi, _mm256_mul_epi32(delta_v, y)),
            );
            k += 4;
        }
    }
    for i in k..n {
        sum_lo[i] += delta * src_lo[i] as i64;
        sum_hi[i] += delta * src_hi[i] as i64;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_batch_avx2<const B: usize>(
    sum_lo: &mut [u64],
    sum_hi: &mut [u64],
    coeff_lo: &[u32; B],
    coeff_hi: &[u32; B],
    records: &[&[u16]; B],
    mask: u16,
) {
    use std::arch::x86_64::*;
    let n = sum_lo.len();
    let full_end = n.saturating_sub(usize::from(mask != u16::MAX));
    let mut a_lo = [_mm256_setzero_si256(); B];
    let mut a_hi = [_mm256_setzero_si256(); B];
    for r in 0..B {
        a_lo[r] = _mm256_set1_epi64x(coeff_lo[r] as i64);
        a_hi[r] = _mm256_set1_epi64x(coeff_hi[r] as i64);
    }
    let mut k = 0;
    // SAFETY: the caller checks equal slice lengths. Each vector access
    // covers four in-bounds symbols; the masked last cell is handled below.
    unsafe {
        while k + 4 <= full_end {
            let mut lo = _mm256_loadu_si256(sum_lo.as_ptr().add(k) as *const _);
            let mut hi = _mm256_loadu_si256(sum_hi.as_ptr().add(k) as *const _);
            for r in 0..B {
                let four = _mm_loadl_epi64(records[r].as_ptr().add(k) as *const _);
                let cell = _mm256_cvtepu16_epi64(four);
                lo = _mm256_add_epi64(lo, _mm256_mul_epu32(a_lo[r], cell));
                hi = _mm256_add_epi64(hi, _mm256_mul_epu32(a_hi[r], cell));
            }
            _mm256_storeu_si256(sum_lo.as_mut_ptr().add(k) as *mut _, lo);
            _mm256_storeu_si256(sum_hi.as_mut_ptr().add(k) as *mut _, hi);
            k += 4;
        }
    }
    while k < n {
        let mask = if k + 1 == n { mask } else { u16::MAX };
        for r in 0..B {
            let cell = (records[r][k] & mask) as u64;
            sum_lo[k] += coeff_lo[r] as u64 * cell;
            sum_hi[k] += coeff_hi[r] as u64 * cell;
        }
        k += 1;
    }
}

/// Bytes of the response-packing tables of a preprocessing: the prepacked
/// columns and the tables of every reply ciphertext.
fn packing_table_bytes(offline: &OfflinePrecomputedValues<'static>) -> usize {
    let poly = |matrix: &PolyMatrixNTT<'static>| matrix.as_slice().len() * 8;
    let prepacked = offline
        .prepacked_lwe
        .iter()
        .flatten()
        .map(poly)
        .sum::<usize>();
    let tables = offline
        .precomp
        .iter()
        .map(|(first, rest, indices)| {
            poly(first)
                + rest
                    .iter()
                    .map(|table| table.as_slice().len() * 8)
                    .sum::<usize>()
                + indices
                    .iter()
                    .map(|list| list.len() * std::mem::size_of::<usize>())
                    .sum::<usize>()
        })
        .sum::<usize>();
    prepacked + tables
}

/// The all-zero blocks of `block` consecutive cells of a column, the unit
/// in which the hint's kernel skips work.
fn count_zero_blocks(column: &[u16], block: usize) -> u64 {
    column
        .chunks_exact(block)
        .filter(|cells| cells.iter().all(|&cell| cell == 0))
        .count() as u64
}

/// A client's secret key for one database shape.
pub struct PirClient {
    shape: PirShape,
    params: &'static Params,
    client: Client<'static>,
}

impl PirClient {
    /// Generate a fresh secret key.
    pub fn generate(shape: PirShape) -> Self {
        let params = shape.params();
        let mut client = Client::init(params);
        client.generate_secret_keys();
        Self {
            shape,
            params,
            client,
        }
    }

    /// The packing keys that accompany every query under this secret key.
    pub fn pack_keys(&self) -> PackKeys {
        PackKeys::from_matrices(
            self.params,
            &factored::client_pack_pub_params(self.params, &self.client),
        )
    }

    /// The query for one pattern.
    pub fn query(&self, pattern: usize) -> Vec<u64> {
        assert!(pattern < self.shape.patterns);
        factored::client_query(self.params, &self.client, pattern)
            .as_slice()
            .to_vec()
    }

    /// A response that decodes to `slots`, the cells of each slot of a row,
    /// encrypted directly under the secret key in the format the server's
    /// packing produces: fresh RLWE ciphertexts, one per instance, switched
    /// to the response moduli. For benchmarks of the client alone: its
    /// decoding runs the same operations on these as on a server's response,
    /// and the server's preprocessing and answer are skipped.
    #[cfg(feature = "bench")]
    pub fn encrypt_slots(&self, slots: &[Vec<u64>]) -> Vec<Vec<u8>> {
        use rand::SeedableRng;
        use spiral_rs::poly::PolyMatrixRaw;
        use ypir::modulus_switch::ModulusSwitch;
        let params = self.params;
        let mut cells = vec![0u64; params.instances * params.poly_len];
        assert_eq!(slots.len(), self.shape.slots);
        for (slot, values) in slots.iter().enumerate() {
            assert_eq!(values.len(), self.shape.slot_width(slot));
            let packing = self.shape.packing(slot);
            cells[packing.base..packing.base + packing.full]
                .copy_from_slice(&values[..packing.full]);
            if packing.tail_bits != 0 {
                let value = values[packing.full] & ((1 << packing.tail_bits) - 1);
                for j in 0..packing.tail_columns() {
                    cells[packing.tail_base + j] |=
                        ((value << packing.shift) >> (j * CELL_BITS)) & ((1 << CELL_BITS) - 1);
                }
            }
        }
        let (q1, q2) = (params.get_q_prime_1(), params.get_q_prime_2());
        let mut rng = rand_chacha::ChaCha20Rng::from_entropy();
        let mut rng_pub = rand_chacha::ChaCha20Rng::from_entropy();
        cells
            .chunks_exact(params.poly_len)
            .map(|coefficients| {
                let mut plaintext = PolyMatrixRaw::zero(params, 1, 1);
                for (place, &cell) in plaintext.data.as_mut_slice().iter_mut().zip(coefficients) {
                    // The decoder rescales from the ciphertext modulus to the
                    // plaintext modulus and rounds.
                    *place = ((cell as u128 * params.modulus as u128
                        + params.pt_modulus as u128 / 2)
                        / params.pt_modulus as u128) as u64;
                }
                self.client
                    .encrypt_matrix_reg(&plaintext.ntt(), &mut rng, &mut rng_pub)
                    .raw()
                    .switch(q1, q2)
            })
            .collect()
    }

    /// Decrypt a response into the cells of each slot of the selected row.
    pub fn decode(&self, response: &[Vec<u8>]) -> Result<Vec<Vec<u64>>, PirError> {
        let params = self.params;
        // One ciphertext per `poly_len` columns, each of two polynomials
        // reduced to `q'_2` and `q'_1`.
        let coefficient_bits = (params.get_q_prime_1() as f64).log2().ceil() as usize
            + (params.get_q_prime_2() as f64).log2().ceil() as usize;
        let ciphertext_bytes = (coefficient_bits * params.poly_len).div_ceil(8);
        if response.len() != params.instances
            || response.iter().any(|ct| ct.len() != ciphertext_bytes)
        {
            return Err(PirError::Response);
        }
        let cells = factored::client_decode(params, &self.client, response);
        Ok((0..self.shape.slots)
            .map(|slot| self.shape.packing(slot).unpack(&cells))
            .collect())
    }
}

/// The table of complete openings: every row of the PIR database stored in
/// full and served with YPIR's response packing.
///
/// It answers exactly as [`PirDatabase`] does and stores each record once per
/// row that uses it. It is the reference of the equality tests and a
/// baseline of the benchmarks.
pub struct TableDatabase {
    params: &'static Params,
    server: YServer<'static, u16>,
    offline: OfflinePrecomputedValues<'static>,
}

impl TableDatabase {
    /// Build an ordinary row database and measure its actual hint and packing
    /// preprocessing. Used by the TreePIR baseline, with the same compact
    /// packing tables as the structured Undine backend. `fill_row` receives
    /// only unpadded rows; the supplied buffer includes padded columns.
    pub fn from_rows_timed(
        shape: PirShape,
        fill_row: impl FnMut(usize, &mut [u16]),
    ) -> (Self, std::time::Duration, PreprocessTimes) {
        let params = shape.params();
        let rows = shape.db_rows();
        let cols = shape.db_cols();
        let now = std::time::Instant::now();
        let server = dense_server_from_rows(shape, fill_row);
        let build = now.elapsed();
        let now = std::time::Instant::now();
        let query = factored::pseudorandom_query(params, 0);
        let mut hint_columns = Vec::with_capacity(cols * params.poly_len);
        let mut kernel = factored::HintColumnKernel::new(params, &query);
        for column in server.db().chunks_exact(rows) {
            kernel.append_column(column, &mut hint_columns);
        }
        let hint = ypir::transpose::transpose_generic(&hint_columns, cols, params.poly_len);
        drop(hint_columns);
        drop(kernel);
        let hint_time = now.elapsed();
        let now = std::time::Instant::now();
        let offline = factored::offline_from_hint(params, hint);
        let packing_time = now.elapsed();
        (
            Self {
                params,
                server,
                offline,
            },
            build,
            PreprocessTimes {
                gather: std::time::Duration::ZERO,
                hint: hint_time,
                packing: packing_time,
            },
        )
    }

    /// Physical bytes of the padded 14-bit-symbol database in u16 storage.
    pub fn database_bytes(&self) -> usize {
        self.server.db().len() * std::mem::size_of::<u16>()
    }

    /// Hint and compact response-packing tables, with the same accounting
    /// used by `PirDatabase::preprocessed_bytes`.
    pub fn preprocessed_bytes(&self) -> usize {
        self.hint_bytes() + self.packing_bytes()
    }

    /// Bytes of the hint: `poly_len` words per database column.
    pub fn hint_bytes(&self) -> usize {
        self.offline.hint_0.len() * 8
    }

    /// Bytes of the response-packing tables.
    pub fn packing_bytes(&self) -> usize {
        packing_table_bytes(&self.offline)
    }

    /// The blocks of `poly_len` rows of one database column that are all
    /// zero, out of all such blocks, as [`PirDatabase::zero_hint_blocks`]
    /// counts them.
    pub fn zero_hint_blocks(&self) -> (u64, u64) {
        let db_rows = factored::simplepir_db_rows(self.params);
        let zero = self
            .server
            .db()
            .chunks_exact(db_rows)
            .map(|column| count_zero_blocks(column, self.params.poly_len))
            .sum();
        let blocks = self.server.db().len() / self.params.poly_len;
        (zero, blocks as u64)
    }

    /// Write out every row of `database` and preprocess the table.
    pub fn from_database(database: &PirDatabase) -> Self {
        let params = database.params;
        let shape = database.shape;
        let db_rows = factored::simplepir_db_rows(params);
        let db_cols = factored::simplepir_db_cols(params);
        // `YServer::new` reads its input row by row.
        let table = (0..db_rows).flat_map(|row| {
            let mut cells = vec![0u16; db_cols];
            if row < shape.patterns {
                for slot in 0..shape.slots {
                    let id = database.rows[row * shape.slots + slot];
                    if id == NO_RECORD {
                        continue;
                    }
                    let source = database.slot_cells(slot, id as usize);
                    let full = shape.slot_bits(slot) / CELL_BITS;
                    let base = shape.slot_offset(slot);
                    cells[base..base + full].copy_from_slice(&source[..full]);
                    // Independent bitwise reference for the suffix.
                    for bit in 0..shape.slot_bits(slot) % CELL_BITS {
                        let dest =
                            shape.full_columns() * CELL_BITS + shape.tail_bit_offset(slot) + bit;
                        cells[dest / CELL_BITS] |=
                            ((source[full] >> bit) & 1) << (dest % CELL_BITS);
                    }
                }
            }
            cells
        });
        let server = YServer::<u16>::new(params, table, true, false, true);
        let offline = server.perform_offline_precomputation_simplepir(None);
        Self {
            params,
            server,
            offline,
        }
    }

    /// The SimplePIR hint of the table.
    pub fn hint_values(&self) -> &[u64] {
        &self.offline.hint_0
    }

    /// The first pass of a query over the table.
    pub fn first_pass(&self, query: &[u64]) -> Result<Vec<u64>, PirError> {
        if query.len() != factored::simplepir_db_rows(self.params) {
            return Err(PirError::Query);
        }
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: the host supports AVX2 and the matrix has one column
            // of query.len() entries per output.
            return Ok(unsafe { dense_first_pass_avx2(self.params, query, self.server.db()) });
        }
        let aligned = self.aligned_query(query)?;
        Ok(self
            .server
            .answer_query(aligned.as_slice())
            .as_slice()
            .to_vec())
    }

    /// Answer one query with the dense scan and YPIR's response packing.
    pub fn answer(&self, query: &[u64], pack_keys: &PackKeys) -> Result<Vec<Vec<u8>>, PirError> {
        let pack_keys = pack_keys.to_matrices(self.params)?;
        let first_pass = self.first_pass(query)?;
        Ok(factored::response_from_first_pass(
            self.params,
            &self.offline,
            &first_pass,
            &pack_keys,
        ))
    }

    fn aligned_query(&self, query: &[u64]) -> Result<AlignedMemory64, PirError> {
        if query.len() != factored::simplepir_db_rows(self.params) {
            return Err(PirError::Query);
        }
        let mut aligned = AlignedMemory64::new(query.len());
        aligned.as_mut_slice().copy_from_slice(query);
        Ok(aligned)
    }
}

/// Pack each source row once, then transpose small tiles into YPIR's
/// column-major database. Direct row-at-a-time writes touch a new page for
/// almost every cell at large dimensions.
fn dense_server_from_rows(
    shape: PirShape,
    mut fill_row: impl FnMut(usize, &mut [u16]),
) -> YServer<'static, u16> {
    let params = shape.params();
    let rows = shape.db_rows();
    let cols = shape.db_cols();
    let mut server = YServer::<u16>::new(params, std::iter::repeat(0), true, true, true);
    let matrix = server.db_mut();
    const ROW_BLOCK: usize = 256;
    const TILE: usize = 32;
    // A cache-line pad avoids mapping every scratch row to the same cache
    // sets when cols is a multiple of the backend's polynomial length.
    let stride = cols + TILE;
    let mut scratch = vec![0u16; ROW_BLOCK * stride];
    for start in (0..shape.patterns).step_by(ROW_BLOCK) {
        let count = (shape.patterns - start).min(ROW_BLOCK);
        scratch.fill(0);
        for row in 0..count {
            fill_row(start + row, &mut scratch[row * stride..row * stride + cols]);
        }
        for col_start in (0..cols).step_by(TILE) {
            for row_start in (0..count).step_by(TILE) {
                let end = (row_start + TILE).min(count);
                for col in col_start..(col_start + TILE).min(cols) {
                    let destination = &mut matrix[col * rows + start..col * rows + start + count];
                    for row in row_start..end {
                        destination[row] = scratch[row * stride + col];
                    }
                }
            }
        }
    }
    server
}

/// Dense YPIR matrix-vector multiplication with the same AVX2 instruction
/// set used by the structured scan. Reducing each 2^15-term block bounds
/// even noncanonical 32-bit query residues times arbitrary u16 cells below
/// 2^63; no lane or horizontal sum can overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dense_first_pass_avx2(params: &Params, query: &[u64], database: &[u16]) -> Vec<u64> {
    use std::arch::x86_64::*;
    const CHUNK: usize = 1 << 15;
    assert!(!query.is_empty());
    assert_eq!(database.len() % query.len(), 0);
    let mut output = Vec::with_capacity(database.len() / query.len());
    for column in database.chunks_exact(query.len()) {
        let mut value = 0;
        for (a, b) in query.chunks(CHUNK).zip(column.chunks(CHUNK)) {
            let mut sum_lo = _mm256_setzero_si256();
            let mut sum_hi = _mm256_setzero_si256();
            let end = a.len() / 4 * 4;
            for i in (0..end).step_by(4) {
                // SAFETY: i..i+4 is in both slices; unaligned loads accept
                // ordinary Vec storage and the u16 load reads only 8 bytes.
                let (coefficients, cells) = unsafe {
                    (
                        _mm256_loadu_si256(a.as_ptr().add(i).cast()),
                        _mm256_cvtepu16_epi64(_mm_loadl_epi64(b.as_ptr().add(i).cast())),
                    )
                };
                sum_lo = _mm256_add_epi64(sum_lo, _mm256_mul_epu32(coefficients, cells));
                sum_hi = _mm256_add_epi64(
                    sum_hi,
                    _mm256_mul_epu32(_mm256_srli_epi64(coefficients, 32), cells),
                );
            }
            let mut lanes_lo = [0u64; 4];
            let mut lanes_hi = [0u64; 4];
            // SAFETY: each local array has space for all four u64 lanes.
            unsafe {
                _mm256_storeu_si256(lanes_lo.as_mut_ptr().cast(), sum_lo);
                _mm256_storeu_si256(lanes_hi.as_mut_ptr().cast(), sum_hi);
            }
            let mut lo = lanes_lo.iter().sum::<u64>();
            let mut hi = lanes_hi.iter().sum::<u64>();
            for (&coefficient, &cell) in a[end..].iter().zip(&b[end..]) {
                lo += (coefficient & LOW_MASK) * u64::from(cell);
                hi += (coefficient >> 32) * u64::from(cell);
            }
            let lo = barrett_coeff_u64(params, lo, 0);
            let hi = barrett_coeff_u64(params, hi, 1);
            value = barrett_u64(params, value + params.crt_compose_2(lo, hi));
        }
        output.push(value);
    }
    output
}

/// Transpose an initialized row tile into the active rows of each column.
/// The scalar edges also handle every dimension on non-x86-64 hosts.
fn transpose_hint_tile(
    tile: &[u16],
    columns: usize,
    rows: usize,
    values: &mut [u16],
    stride: usize,
    start: usize,
    vectorized: bool,
) {
    assert!(columns > 0 && stride > 0);
    assert!(rows <= tile.len() / columns);
    assert!(columns <= values.len() / stride);
    assert!(start <= stride && rows <= stride - start);
    let vector_columns = if cfg!(target_arch = "x86_64") && vectorized {
        columns / 8 * 8
    } else {
        0
    };
    let vector_rows = rows / 8 * 8;
    for column in (0..vector_columns).step_by(8) {
        for row in (0..vector_rows).step_by(8) {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: the assertions bound both slices. These loops select
            // complete 8x8 blocks; SSE2 is mandatory on x86-64, and the
            // source and destination are disjoint borrowed slices.
            unsafe {
                transpose_hint_8x8(
                    tile.as_ptr().add(row * columns + column),
                    columns,
                    values.as_mut_ptr().add(column * stride + start + row),
                    stride,
                );
            }
        }
        for col in column..column + 8 {
            for row in vector_rows..rows {
                values[col * stride + start + row] = tile[row * columns + col];
            }
        }
    }
    for col in vector_columns..columns {
        for row in 0..rows {
            values[col * stride + start + row] = tile[row * columns + col];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn transpose_hint_8x8(src: *const u16, src_stride: usize, dst: *mut u16, dst_stride: usize) {
    use std::arch::x86_64::*;
    // SAFETY: the caller supplies eight valid rows of eight u16s at each
    // stride. Unaligned loads/stores impose no additional alignment rule.
    unsafe {
        let r0 = _mm_loadu_si128(src.cast());
        let r1 = _mm_loadu_si128(src.add(src_stride).cast());
        let r2 = _mm_loadu_si128(src.add(2 * src_stride).cast());
        let r3 = _mm_loadu_si128(src.add(3 * src_stride).cast());
        let r4 = _mm_loadu_si128(src.add(4 * src_stride).cast());
        let r5 = _mm_loadu_si128(src.add(5 * src_stride).cast());
        let r6 = _mm_loadu_si128(src.add(6 * src_stride).cast());
        let r7 = _mm_loadu_si128(src.add(7 * src_stride).cast());
        let t0 = _mm_unpacklo_epi16(r0, r1);
        let t1 = _mm_unpackhi_epi16(r0, r1);
        let t2 = _mm_unpacklo_epi16(r2, r3);
        let t3 = _mm_unpackhi_epi16(r2, r3);
        let t4 = _mm_unpacklo_epi16(r4, r5);
        let t5 = _mm_unpackhi_epi16(r4, r5);
        let t6 = _mm_unpacklo_epi16(r6, r7);
        let t7 = _mm_unpackhi_epi16(r6, r7);
        let u0 = _mm_unpacklo_epi32(t0, t2);
        let u1 = _mm_unpackhi_epi32(t0, t2);
        let u2 = _mm_unpacklo_epi32(t1, t3);
        let u3 = _mm_unpackhi_epi32(t1, t3);
        let u4 = _mm_unpacklo_epi32(t4, t6);
        let u5 = _mm_unpackhi_epi32(t4, t6);
        let u6 = _mm_unpacklo_epi32(t5, t7);
        let u7 = _mm_unpackhi_epi32(t5, t7);
        let out = [
            _mm_unpacklo_epi64(u0, u4),
            _mm_unpackhi_epi64(u0, u4),
            _mm_unpacklo_epi64(u1, u5),
            _mm_unpackhi_epi64(u1, u5),
            _mm_unpacklo_epi64(u2, u6),
            _mm_unpackhi_epi64(u2, u6),
            _mm_unpacklo_epi64(u3, u7),
            _mm_unpackhi_epi64(u3, u7),
        ];
        for (column, value) in out.into_iter().enumerate() {
            _mm_storeu_si128(dst.add(column * dst_stride).cast(), value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hint_tile_transpose_matches_scalar_with_partial_tiles() {
        for columns in [1, 7, 8, 9, 31, 32, 33, 128, 1024] {
            for rows in [1, 7, 8, 9, 31, 32] {
                let input = (0..rows * columns)
                    .map(|i| (i * 97 + 13) as u16)
                    .collect::<Vec<_>>();
                let stride = rows + 40;
                let mut scalar = vec![u16::MAX; columns * stride];
                let mut vector = scalar.clone();
                transpose_hint_tile(&input, columns, rows, &mut scalar, stride, 3, false);
                transpose_hint_tile(&input, columns, rows, &mut vector, stride, 3, true);
                assert_eq!(vector, scalar, "columns={columns}, rows={rows}");
                for col in 0..columns {
                    assert_eq!(&vector[col * stride..][..3], &[u16::MAX; 3]);
                    for row in 0..rows {
                        assert_eq!(vector[col * stride + 3 + row], input[row * columns + col]);
                    }
                    assert!(
                        vector[col * stride + 3 + rows..(col + 1) * stride]
                            .iter()
                            .all(|&value| value == u16::MAX)
                    );
                }
            }
        }
    }

    #[test]
    fn packed_suffix_round_trips_every_residual_width() {
        for record_bits in (1..=2 * CELL_BITS).chain([256, 512, 8192]) {
            let shape = PirShape {
                patterns: 1,
                slots: 19,
                group_slots: 16,
                record_bits,
                record_cells: record_bits.div_ceil(CELL_BITS),
            };
            let slots = (0..shape.slots)
                .map(|slot| {
                    let bits = shape.slot_bits(slot);
                    let mut data = (0..shape.slot_width(slot))
                        .map(|i| ((i * 97 + slot * 733 + 13) & ((1 << CELL_BITS) - 1)) as u16)
                        .collect::<Vec<_>>();
                    if bits % CELL_BITS != 0 {
                        *data.last_mut().unwrap() &= (1 << (bits % CELL_BITS)) - 1;
                    }
                    data
                })
                .collect::<Vec<_>>();
            let mut packed = vec![0u64; shape.row_cells()];
            let mut expected_bits = Vec::new();
            for (slot, data) in slots.iter().enumerate() {
                let full = shape.slot_bits(slot) / CELL_BITS;
                for value in &data[..full] {
                    expected_bits.extend((0..CELL_BITS).map(|b| (value >> b) & 1));
                }
                let packing = shape.packing(slot);
                for offset in 0..packing.columns() {
                    packed[packing.column(offset)] += packing.cell(data, offset) as u64;
                }
            }
            for (slot, data) in slots.iter().enumerate() {
                let bits = shape.slot_bits(slot);
                expected_bits
                    .extend((0..bits % CELL_BITS).map(|b| (data[bits / CELL_BITS] >> b) & 1));
            }
            let expected = expected_bits
                .chunks(CELL_BITS)
                .map(|bits| {
                    bits.iter()
                        .enumerate()
                        .map(|(i, &bit)| (bit as u64) << i)
                        .sum::<u64>()
                })
                .collect::<Vec<_>>();
            assert_eq!(packed, expected, "record_bits={record_bits}");
            assert_eq!(
                shape.row_cells(),
                (0..shape.slots)
                    .map(|s| shape.slot_bits(s))
                    .sum::<usize>()
                    .div_ceil(CELL_BITS)
            );
            for (slot, original) in slots.iter().enumerate() {
                assert_eq!(
                    shape.packing(slot).unpack(&packed),
                    original.iter().map(|&x| x as u64).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn height_32_record_fits_37_outputs_without_slot_padding() {
        let shape = PirShape {
            patterns: 65536,
            slots: 145,
            group_slots: 129,
            record_bits: 32 * 256,
            record_cells: (32usize * 256).div_ceil(CELL_BITS),
        };
        assert_eq!(shape.row_cells(), 75776);
        assert_eq!(shape.db_cols(), 75776);
        assert_eq!(shape.outputs(), 37);
    }

    #[test]
    fn batched_columns_match_scalar_gathers_across_slots_and_padding() {
        for patterns in [1, 15, 16, 17, 31, 32, 33, 97, 2049] {
            let shape = PirShape {
                patterns,
                slots: 4,
                group_slots: 2,
                record_cells: 37,
                record_bits: 37 * CELL_BITS - 6,
            };
            let cells = (0..11 * shape.record_cells)
                .map(|i| ((i * 97 + 3) % (1 << CELL_BITS)) as u16)
                .collect::<Vec<_>>();
            let upper = [7, 5]
                .into_iter()
                .enumerate()
                .map(|(level, count)| {
                    (0..count * UPPER_CELLS)
                        .map(|i| ((i * 73 + 101 * level + 9) % (1 << CELL_BITS)) as u16)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let rows = (0..patterns)
                .flat_map(|pattern| {
                    [11, 11, 7, 5]
                        .into_iter()
                        .enumerate()
                        .map(move |(slot, count)| {
                            if (pattern + slot) % 5 == 0 {
                                NO_RECORD
                            } else {
                                ((pattern * 3 + slot) % count) as u32
                            }
                        })
                })
                .collect::<Vec<_>>();
            // Cross row tiles, both kinds of slot, empty/repeated records,
            // padding rows/columns, and a short final column batch.
            for start in [0, 1, 31, 36, 64, 73, 93, 108, 111, 112, 2039] {
                let cols = start..(start + HINT_COLUMN_BATCH).min(shape.db_cols());
                for stride in [shape.db_rows(), shape.db_rows() + HINT_COLUMN_PADDING] {
                    let mut actual = vec![u16::MAX; cols.len() * stride];
                    PirDatabase::fill_columns(
                        &shape,
                        &cells,
                        &upper,
                        HintRowIds::by_pattern(&rows, &shape),
                        cols.clone(),
                        &mut actual,
                    );
                    let mut expected = vec![u16::MAX; stride];
                    for (index, col) in cols.clone().enumerate() {
                        PirDatabase::fill_column(&shape, &cells, &upper, &rows, col, &mut expected);
                        assert_eq!(
                            &actual[index * stride..][..stride],
                            &expected,
                            "patterns={patterns}, column={col}, stride={stride}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gather_reuses_batches_across_active_columns_in_both_id_orders() {
        let shape = PirShape {
            patterns: 33,
            slots: 5,
            group_slots: 3,
            record_cells: 600,
            record_bits: 600 * CELL_BITS - 3,
        };
        let cells = (0..7 * shape.record_cells)
            .map(|i| ((i * 97 + 3) % (1 << CELL_BITS)) as u16)
            .collect::<Vec<_>>();
        let upper = [3, 2]
            .into_iter()
            .map(|count| {
                (0..count * UPPER_CELLS)
                    .map(|i| ((i * 73 + 9) % (1 << CELL_BITS)) as u16)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rows = (0..shape.patterns)
            .flat_map(|pattern| {
                [7, 7, 7, 3, 2]
                    .into_iter()
                    .enumerate()
                    .map(move |(slot, count)| {
                        if (pattern + slot) % 5 == 0 {
                            NO_RECORD
                        } else {
                            ((pattern * 3 + slot) % count) as u32
                        }
                    })
            })
            .collect::<Vec<_>>();
        let mut by_slot = vec![0; rows.len()];
        for pattern in 0..shape.patterns {
            for slot in 0..shape.slots {
                by_slot[slot * shape.patterns + pattern] = rows[pattern * shape.slots + slot];
            }
        }
        let stride = shape.db_rows() + HINT_COLUMN_PADDING;
        let mut actual = vec![u16::MAX; HINT_COLUMN_BATCH * stride];
        let mut expected = vec![0; stride];
        for ids in [
            HintRowIds::by_pattern(&rows, &shape),
            HintRowIds::by_slot(&by_slot, &shape),
        ] {
            for start in [0, HINT_COLUMN_BATCH, 601, 1793, 2041] {
                let cols = start..(start + HINT_COLUMN_BATCH).min(shape.db_cols());
                PirDatabase::fill_columns(
                    &shape,
                    &cells,
                    &upper,
                    ids,
                    cols.clone(),
                    &mut actual[..cols.len() * stride],
                );
                for (index, col) in cols.enumerate() {
                    PirDatabase::fill_column(&shape, &cells, &upper, &rows, col, &mut expected);
                    assert_eq!(
                        &actual[index * stride..][..stride],
                        &expected,
                        "start={start}, col={col}, pattern_stride={}",
                        ids.pattern_stride
                    );
                }
            }
        }
    }

    #[test]
    fn batched_hint_matches_scalar_hint_with_multiple_workers() {
        // Three workers split the third group slot; the last worker gets
        // only padding. The row count crosses a YPIR polynomial boundary.
        let shape = PirShape {
            patterns: 2049,
            slots: 5,
            group_slots: 3,
            record_cells: 257,
            record_bits: 257 * CELL_BITS - 3,
        };
        let cells = (0..13 * shape.record_cells)
            .map(|i| ((i * 97 + 3) % (1 << CELL_BITS)) as u16)
            .collect::<Vec<_>>();
        let upper = [7, 5]
            .into_iter()
            .map(|count| {
                (0..count * UPPER_CELLS)
                    .map(|i| ((i * 73 + count) % (1 << CELL_BITS)) as u16)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rows = (0..shape.patterns)
            .flat_map(|pattern| {
                [13, 13, 13, 7, 5]
                    .into_iter()
                    .enumerate()
                    .map(move |(slot, count)| {
                        if (pattern + slot) % 5 == 0 {
                            NO_RECORD
                        } else {
                            ((pattern * 3 + slot) % count) as u32
                        }
                    })
            })
            .collect::<Vec<_>>();
        let params = shape.params();
        let query = factored::pseudorandom_query(params, SEED_0);
        let expected =
            factored::hint_columns_from_columns(params, &query, 0..shape.db_cols(), |col, out| {
                PirDatabase::fill_column(&shape, &cells, &upper, &rows, col, out)
            });
        for threads in [1, 3] {
            assert_eq!(
                PirDatabase::hint(&shape, params, &cells, &upper, &rows, threads),
                expected,
                "hint with {threads} workers"
            );
        }
    }

    #[test]
    fn tiled_dense_table_matches_original_ypir() {
        // Cross both tile dimensions, a row-block boundary, and the final
        // unpadded row/column. Also check the zero padding explicitly.
        let shape = PirShape {
            patterns: 259,
            slots: 1,
            group_slots: 1,
            record_cells: 2051,
            record_bits: 2051 * CELL_BITS,
        };
        let cell = |row: usize, col: usize| ((row * 73 + col * 97) % (1 << CELL_BITS)) as u16;
        let (table, _, _) = TableDatabase::from_rows_timed(shape, |row, out| {
            for (col, value) in out[..shape.record_cells].iter_mut().enumerate() {
                *value = cell(row, col);
            }
        });
        for (col, column) in table.server.db().chunks_exact(shape.db_rows()).enumerate() {
            for (row, &value) in column.iter().enumerate() {
                let expected = if row < shape.patterns && col < shape.record_cells {
                    cell(row, col)
                } else {
                    0
                };
                assert_eq!(value, expected, "row {row}, col {col}");
            }
        }
        let client = PirClient::generate(shape);
        let keys = client.pack_keys();
        let matrices = keys.to_matrices(table.params).unwrap();
        for query in [
            client.query(shape.patterns - 1),
            vec![u64::MAX; shape.db_rows()],
        ] {
            let aligned = table.aligned_query(&query).unwrap();
            assert_eq!(
                table.first_pass(&query).unwrap(),
                table.server.answer_query(aligned.as_slice()).as_slice(),
            );
            assert_eq!(
                table.answer(&query, &keys).unwrap(),
                table.server.perform_online_computation_simplepir(
                    aligned.as_slice(),
                    &table.offline,
                    &[matrices.as_slice()],
                    None,
                ),
            );
        }
        let response = table
            .answer(&client.query(shape.patterns - 1), &keys)
            .unwrap();
        let plaintext = client.decode(&response).unwrap();
        assert_eq!(
            plaintext[0],
            (0..shape.record_cells)
                .map(|col| u64::from(cell(shape.patterns - 1, col)))
                .collect::<Vec<_>>()
        );
        assert_eq!(table.first_pass(&[0]), Err(PirError::Query));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dense_avx2_matches_integer_arithmetic_across_reduction_boundaries() {
        if !has_avx2() {
            return;
        }
        let params = params_for_scenario_simplepir(2048, 2048 * CELL_BITS);
        for rows in [1, 3, 4, 32768, 32769, 65539] {
            let database = (0..rows * 5)
                .map(|i| u16::MAX - (i % 31) as u16)
                .collect::<Vec<_>>();
            for query in [
                vec![u64::MAX; rows],
                (0..rows)
                    .map(|i| {
                        (i as u64 + 1)
                            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                            .rotate_left((i % 64) as u32)
                    })
                    .collect(),
            ] {
                let expected = database
                    .chunks_exact(rows)
                    .map(|column| {
                        let lo = query
                            .iter()
                            .zip(column)
                            .map(|(&a, &b)| (a & LOW_MASK) as u128 * b as u128)
                            .sum::<u128>()
                            % params.moduli[0] as u128;
                        let hi = query
                            .iter()
                            .zip(column)
                            .map(|(&a, &b)| (a >> 32) as u128 * b as u128)
                            .sum::<u128>()
                            % params.moduli[1] as u128;
                        params.crt_compose_2(lo as u64, hi as u64) % params.modulus
                    })
                    .collect::<Vec<_>>();
                // SAFETY: checked above; dimensions satisfy the kernel's contract.
                assert_eq!(
                    unsafe { dense_first_pass_avx2(&params, &query, &database) },
                    expected
                );
            }
        }
    }

    /// The finalize of a patched column equals Barrett reduction and the
    /// backend's CRT composition, on both kernels and at the bounds of the
    /// sums.
    #[test]
    fn finalize_matches_barrett_and_crt() {
        let shape = PirShape {
            patterns: 1,
            slots: 1,
            group_slots: 1,
            record_cells: 1,
            record_bits: 1 * CELL_BITS,
        };
        let params = shape.params();
        let finalize = Finalize::of(params);
        let (q0, q1) = (params.moduli[0], params.moduli[1]);
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let n = 4 * 1000 + 3;
        let bound = 1i64 << 57;
        let mut sum_lo = (0..n)
            .map(|_| (next() % (2 * bound as u64)) as i64 - bound)
            .collect::<Vec<_>>();
        let mut sum_hi = (0..n)
            .map(|_| (next() % (2 * bound as u64)) as i64 - bound)
            .collect::<Vec<_>>();
        let edges: [i64; 8] = [
            0,
            1,
            -1,
            bound - 1,
            1 - bound,
            q0 as i64,
            -(q1 as i64),
            (q0 * q1) as i64,
        ];
        for (i, &edge) in edges.iter().enumerate() {
            sum_lo[i] = edge;
            sum_hi[edges.len() - 1 - i] = edge;
        }
        let column = (0..n).map(|_| next() % params.modulus).collect::<Vec<_>>();
        let expected = column
            .iter()
            .zip(&sum_lo)
            .zip(&sum_hi)
            .map(|((&value, &lo), &hi)| {
                let lo = lo.rem_euclid(q0 as i64) as u64;
                let hi = hi.rem_euclid(q1 as i64) as u64;
                let add = barrett_u64(params, params.crt_compose_2(lo, hi));
                (value + add) % params.modulus
            })
            .collect::<Vec<_>>();
        let mut scalar = column.clone();
        finalize.add_sums_to_column_scalar(&sum_lo, &sum_hi, &mut scalar);
        assert_eq!(scalar, expected, "scalar finalize");
        let mut both = column.clone();
        finalize.add_sums_to_column(&sum_lo, &sum_hi, &mut both);
        assert_eq!(both, expected, "finalize on this host");
        finalize.reduce_sums(&mut sum_lo, &mut sum_hi);
        for (&lo, &hi) in sum_lo.iter().zip(&sum_hi) {
            assert!(0 <= lo && (lo as u64) < q0);
            assert!(0 <= hi && (hi as u64) < q1);
        }
        let mut reduced = column.clone();
        finalize.add_sums_to_column(&sum_lo, &sum_hi, &mut reduced);
        assert_eq!(
            reduced, expected,
            "finalize after an intermediate reduction"
        );
    }

    #[test]
    fn cached_parameters_match_backend_at_padding_boundaries() {
        // Both dimensions change at a multiple of 2048; the row dimension
        // also rounds up to a power of two. Compare all fields, not just
        // the dimensions used as the cache key.
        for patterns in [1, 2047, 2048, 2049, 4095, 4096, 4097, 8192, 8193] {
            for record_cells in [1, 2047, 2048, 2049, 4095, 4096, 4097] {
                let shape = PirShape {
                    patterns,
                    slots: 1,
                    group_slots: 1,
                    record_cells,
                    record_bits: record_cells * CELL_BITS,
                };
                let expected = params_for_scenario_simplepir(patterns, record_cells * CELL_BITS);
                let actual = shape.params();
                assert!(
                    actual == &expected,
                    "patterns={patterns}, cells={record_cells}"
                );
                assert!(std::ptr::eq(actual, shape.params()));

                // Shapes that differ only in padding share the allocation.
                let padded = PirShape {
                    patterns: factored::simplepir_db_rows(actual),
                    record_cells: factored::simplepir_db_cols(actual),
                    record_bits: factored::simplepir_db_cols(actual) * CELL_BITS,
                    ..shape
                };
                assert!(std::ptr::eq(actual, padded.params()));
            }
        }
    }

    #[test]
    fn pair_folding_matches_independent_range_sums() {
        let params = params_for_scenario_simplepir(2048, 2048 * CELL_BITS);
        for len in [1, 2, 3, 7, 8, 9, 45, 1025] {
            let original = (0..len)
                .map(|i| {
                    // Include values near both moduli, zero residues, and
                    // different values in each half of the packed word.
                    let lo = (params.moduli[0] - 1) * (i as u64 % 5) % params.moduli[0];
                    let hi = (params.moduli[1] - 1) * (i as u64 % 7) % params.moduli[1];
                    lo | (hi << 32)
                })
                .collect::<Vec<_>>();
            let mut folded = original.clone();
            let mut width = 1;
            while folded.len() > 1 {
                fold_query_pairs(&params, &mut folded);
                width *= 2;
                for (sum, block) in folded.iter().zip(original.chunks(width)) {
                    let lo = block.iter().map(|v| (v & LOW_MASK) as u128).sum::<u128>()
                        % params.moduli[0] as u128;
                    let hi = block.iter().map(|v| (v >> 32) as u128).sum::<u128>()
                        % params.moduli[1] as u128;
                    assert_eq!(*sum, lo as u64 | ((hi as u64) << 32));
                }
                assert_eq!(folded.len(), original.len().div_ceil(width));
            }
        }
    }

    #[test]
    fn batched_scan_matches_scalar_with_partial_last_cells() {
        // Exercise both dispatch paths, including widths around SIMD edges,
        // zero-weight lanes in a partial batch, and maximal u32 coefficients.
        for width in (0..=20).chain([585, 586]) {
            let data = std::array::from_fn::<_, 4, _>(|r| {
                (0..width)
                    .map(|k| u16::MAX - ((17 * k + r) % 257) as u16)
                    .collect::<Vec<_>>()
            });
            let records = data.each_ref().map(Vec::as_slice);
            let coeff_lo = [u32::MAX, 0, u32::MAX - 1, 37];
            let coeff_hi = [0, u32::MAX, 17, u32::MAX - 2];
            for mask in [1, 3, 255, 8191, u16::MAX] {
                let mut lo = vec![123u64; width];
                let mut hi = vec![456u64; width];
                let mut expected_lo = lo.clone();
                let mut expected_hi = hi.clone();
                add_scaled_batch_scalar(
                    &mut expected_lo,
                    &mut expected_hi,
                    &coeff_lo,
                    &coeff_hi,
                    &records,
                    mask,
                );
                add_scaled_batch(&mut lo, &mut hi, &coeff_lo, &coeff_hi, &records, mask);
                assert_eq!(
                    (lo, hi),
                    (expected_lo, expected_hi),
                    "width={width}, mask={mask}"
                );
            }
        }
    }

    #[test]
    fn first_pass_reduces_group_slots_independently() {
        // Cross two 2^15-product reduction boundaries in slot 0 and one
        // in slot 1. Five cells exercise the AVX2 block and scalar tail.
        // Preprocessing is deliberately absent: first_pass never reads it.
        for record_bits in [5 * CELL_BITS, 5 * CELL_BITS - 6] {
            let shape = PirShape {
                patterns: (1 << 16) + 1,
                slots: 2,
                group_slots: 2,
                record_cells: 5,
                record_bits,
            };
            let params = shape.params();
            let records = 2 * shape.patterns;
            let cells = (0..records * shape.record_cells)
                .map(|i| u16::MAX - (i % 31) as u16)
                .collect::<Vec<_>>();
            let rows = (0..shape.patterns)
                .flat_map(|pattern| {
                    [
                        (2 * pattern) as u32,
                        if pattern % 3 == 0 {
                            NO_RECORD
                        } else {
                            (2 * pattern + 1) as u32
                        },
                    ]
                })
                .collect::<Vec<_>>();
            let database = PirDatabase {
                shape,
                params,
                cells,
                records,
                upper: Vec::new(),
                rows,
                hint_cols: Vec::new(),
                offline: OfflinePrecomputedValues {
                    hint_0: Vec::new(),
                    hint_1: Vec::new(),
                    pseudorandom_query_1: Vec::new(),
                    y_constants: (Vec::new(), Vec::new()),
                    smaller_server: None,
                    prepacked_lwe: Vec::new(),
                    fake_pack_pub_params: Vec::new(),
                    precomp: Vec::new(),
                },
                table_threads: 1,
            };
            // Maximal noncanonical query residues and full u16 cells test the
            // documented arithmetic bounds, beyond the ordinary 14-bit cells.
            let query = vec![u64::MAX; factored::simplepir_db_rows(params)];
            let actual = database.first_pass(&query).expect("first pass");
            let mut expected = vec![0u64; actual.len()];
            let mut sums = vec![0u128; shape.row_cells()];
            for row in database.rows.chunks_exact(shape.slots) {
                for (slot, &record) in row.iter().enumerate() {
                    if record == NO_RECORD {
                        continue;
                    }
                    let source = database.record_cells(record as usize);
                    let full = shape.slot_bits(slot) / CELL_BITS;
                    for j in 0..full {
                        sums[shape.slot_offset(slot) + j] += u32::MAX as u128 * source[j] as u128;
                    }
                    for bit in 0..shape.slot_bits(slot) % CELL_BITS {
                        let dest =
                            shape.full_columns() * CELL_BITS + shape.tail_bit_offset(slot) + bit;
                        let value = ((source[full] >> bit) & 1) as u128;
                        sums[dest / CELL_BITS] += u32::MAX as u128 * (value << (dest % CELL_BITS));
                    }
                }
            }
            for (col, sum) in sums.into_iter().enumerate() {
                let lo = (sum % params.moduli[0] as u128) as u64;
                let hi = (sum % params.moduli[1] as u128) as u64;
                expected[col] = barrett_u64(params, params.crt_compose_2(lo, hi));
            }
            assert_eq!(actual, expected, "record_bits={record_bits}");
        }
    }

    #[test]
    fn entries_written_one_by_one_give_the_cells_of_a_record() {
        let entries = (0..9u64)
            .map(|i| decode_value([(17 * i + 3) as u8; 32]) + Fp::from(i * i))
            .collect::<Vec<_>>();
        let record_cells = (entries.len() * ENTRY_BYTES * 8).div_ceil(CELL_BITS);
        let mut packed = vec![0u16; record_cells];
        entries_to_cells(&entries, &mut packed);
        let mut written = vec![0u16; record_cells];
        // Any order gives the same cells.
        for index in [4usize, 0, 8, 1, 7, 2, 6, 3, 5] {
            or_entry_into_cells(&mut written, index, entries[index]);
        }
        assert_eq!(written, packed);
        let as_u64 = packed.iter().map(|&cell| cell as u64).collect::<Vec<_>>();
        for (index, entry) in entries.iter().enumerate() {
            assert_eq!(entry_from_cells(&as_u64, index), *entry);
        }
    }
    /// Build only the data used by a scan: no hint or response preprocessing.
    /// Full cells deliberately range over u16; partial cells have nonzero
    /// padding, so a scan must mask their unused bits before multiplying.
    fn synthetic_scan_database(
        record_bits: usize,
        patterns: usize,
        group_slots: usize,
        upper_slots: usize,
    ) -> PirDatabase {
        let shape = PirShape {
            patterns,
            slots: group_slots + upper_slots,
            group_slots,
            record_bits,
            record_cells: record_bits.div_ceil(CELL_BITS),
        };
        let records = patterns * group_slots;
        let cells = (0..records * shape.record_cells)
            .map(|i| (i as u16).wrapping_mul(137).wrapping_add(0xf123))
            .collect::<Vec<_>>();
        let upper = (0..upper_slots)
            .map(|level| {
                let count = patterns.div_ceil(1usize << level);
                (0..count * UPPER_CELLS)
                    .map(|i| {
                        (i as u16)
                            .wrapping_mul(257)
                            .wrapping_add(0xe357 + level as u16)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rows = (0..patterns)
            .flat_map(|pattern| {
                (0..shape.slots).map(move |slot| {
                    if slot < group_slots {
                        if slot != 0 && (pattern + 3 * slot) % 11 == 7 {
                            NO_RECORD
                        } else {
                            // Some groups are shared by two patterns. Every group
                            // still occupies the slot given by its id modulo G.
                            let source = if slot % 3 == 0 { pattern / 2 } else { pattern };
                            (source * group_slots + slot) as u32
                        }
                    } else {
                        let level = slot - group_slots;
                        let sibling = (pattern >> level) ^ 1;
                        if sibling < patterns.div_ceil(1usize << level) {
                            sibling as u32
                        } else {
                            NO_RECORD
                        }
                    }
                })
            })
            .collect();
        PirDatabase {
            shape,
            params: shape.params(),
            cells,
            records,
            upper,
            rows,
            hint_cols: Vec::new(),
            table_threads: 1,
            offline: OfflinePrecomputedValues {
                hint_0: Vec::new(),
                hint_1: Vec::new(),
                pseudorandom_query_1: Vec::new(),
                y_constants: (Vec::new(), Vec::new()),
                smaller_server: None,
                prepacked_lwe: Vec::new(),
                fake_pack_pub_params: Vec::new(),
                precomp: Vec::new(),
            },
        }
    }

    /// Independent row-by-row integer oracle: copy full symbols and place
    /// each residual bit individually. Do not use SlotPacking or scan helpers.
    fn assert_scan_matches_bitwise_rows(database: &PirDatabase, query: &[u64]) {
        let shape = database.shape;
        let widths = (0..shape.slots)
            .map(|slot| {
                if slot < shape.group_slots {
                    shape.record_bits
                } else {
                    ENTRY_BYTES * 8
                }
            })
            .collect::<Vec<_>>();
        let full_columns = widths.iter().map(|bits| bits / CELL_BITS).sum::<usize>();
        let columns = widths.iter().sum::<usize>().div_ceil(CELL_BITS);
        let mut sums_lo = vec![0u128; columns];
        let mut sums_hi = vec![0u128; columns];
        for (pattern, row) in database.rows.chunks_exact(shape.slots).enumerate() {
            let coefficient_lo = (query[pattern] & LOW_MASK) as u128;
            let coefficient_hi = (query[pattern] >> 32) as u128;
            let mut base = 0;
            let mut tail_bit = 0;
            for (slot, (&id, &bits)) in row.iter().zip(&widths).enumerate() {
                let full = bits / CELL_BITS;
                let residual = bits % CELL_BITS;
                if id != NO_RECORD {
                    let source = if slot < shape.group_slots {
                        &database.cells[id as usize * shape.record_cells..][..shape.record_cells]
                    } else {
                        &database.upper[slot - shape.group_slots][id as usize * UPPER_CELLS..]
                            [..UPPER_CELLS]
                    };
                    for (offset, &cell) in source[..full].iter().enumerate() {
                        sums_lo[base + offset] += coefficient_lo * cell as u128;
                        sums_hi[base + offset] += coefficient_hi * cell as u128;
                    }
                    for bit in 0..residual {
                        let dest = tail_bit + bit;
                        let value = (((source[full] >> bit) & 1) as u128) << (dest % CELL_BITS);
                        sums_lo[full_columns + dest / CELL_BITS] += coefficient_lo * value;
                        sums_hi[full_columns + dest / CELL_BITS] += coefficient_hi * value;
                    }
                }
                base += full;
                tail_bit += residual;
            }
        }
        let params = database.params;
        let mut expected = vec![0u64; factored::simplepir_db_cols(params)];
        for col in 0..columns {
            let lo = (sums_lo[col] % params.moduli[0] as u128) as u64;
            let hi = (sums_hi[col] % params.moduli[1] as u128) as u64;
            expected[col] = barrett_u64(params, params.crt_compose_2(lo, hi));
        }
        assert_eq!(
            database.first_pass(query).expect("first pass"),
            expected,
            "bits={}, groups={}, patterns={}, upper={}",
            shape.record_bits,
            shape.group_slots,
            shape.patterns,
            shape.upper_slots()
        );
    }

    #[test]
    fn first_pass_packs_all_residual_widths_and_odd_upper_levels() {
        // Fifteen slots cover every residual phase modulo 14. Sizes below
        // 14 have no full symbols; the extra sizes straddle SIMD widths.
        // Pattern counts leave one, two, or three records in the final
        // batch, and include odd sibling counts at several levels.
        let cases = (1..=28)
            .chain([41, 42, 43, 55, 56, 57, 69, 70, 71, 256, 512, 8192])
            .map(|bits| (bits, 65))
            .chain([(64, 66), (64, 67)]);
        for (record_bits, patterns) in cases {
            let database = synthetic_scan_database(record_bits, patterns, 15, 8);
            let params = database.params;
            let query = (0..factored::simplepir_db_rows(params))
                .map(|i| match i % 17 {
                    0 => 0,
                    1 => u64::MAX,
                    2 => params.moduli[0] | (params.moduli[1] << 32),
                    _ => (i as u64 + 1)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .rotate_left((i % 64) as u32),
                })
                .collect::<Vec<_>>();
            assert_scan_matches_bitwise_rows(&database, &query);
        }
    }

    #[test]
    fn first_pass_reduces_split_upper_tails() {
        // A 12-bit group tail leaves two bits in the first suffix column;
        // the four-bit upper tail splits across that boundary. The upper
        // scan crosses two reductions and has a partial final batch.
        let database = synthetic_scan_database(26, (1 << 16) + 3, 1, 1);
        let query = vec![0xffff_fffd_ffff_ffff; factored::simplepir_db_rows(database.params)];
        assert_scan_matches_bitwise_rows(&database, &query);
    }
}
