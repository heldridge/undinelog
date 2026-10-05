//! Private openings over YPIR's SimplePIR variant.
//!
//! The PIR database has one row per pattern. Row `r` is the concatenation,
//! slot by slot, of the records in `PrivateLayout::row(r)`, each written as
//! plaintext cells of [`CELL_BITS`] bits. A maintainer keeps every record
//! once and never builds the rows:
//! - per query, it adds the query coefficients of all patterns that place a
//!   record in a slot, then multiplies the sum with the stored record;
//! - its preprocessing reads each database column from the stored records.
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
    /// Plaintext cells per group, the width of a group slot.
    pub record_cells: usize,
}

impl PirShape {
    /// Shape of the database of `layout`.
    pub fn of(layout: &PrivateLayout) -> Self {
        Self {
            patterns: layout.patterns(),
            slots: layout.slots(),
            group_slots: layout.group_slots(),
            record_cells: (layout.entries_per_record() * ENTRY_BYTES * 8).div_ceil(CELL_BITS),
        }
    }

    /// Slots above the pattern's subtree.
    pub fn upper_slots(&self) -> usize {
        self.slots - self.group_slots
    }

    /// Plaintext cells of one slot.
    pub fn slot_width(&self, slot: usize) -> usize {
        if slot < self.group_slots { self.record_cells } else { UPPER_CELLS }
    }

    /// First database column of one slot.
    pub fn slot_offset(&self, slot: usize) -> usize {
        if slot < self.group_slots {
            slot * self.record_cells
        } else {
            self.group_slots * self.record_cells + (slot - self.group_slots) * UPPER_CELLS
        }
    }

    /// The slot and the offset within it of one database column, or `None`
    /// for a padding column.
    pub fn slot_of_column(&self, col: usize) -> Option<(usize, usize)> {
        if col < self.group_slots * self.record_cells {
            Some((col / self.record_cells, col % self.record_cells))
        } else {
            let upper = col - self.group_slots * self.record_cells;
            let slot = self.group_slots + upper / UPPER_CELLS;
            (slot < self.slots).then_some((slot, upper % UPPER_CELLS))
        }
    }

    /// Plaintext cells of one record before padding.
    pub fn row_cells(&self) -> usize {
        self.group_slots * self.record_cells + self.upper_slots() * UPPER_CELLS
    }

    /// YPIR parameters for this shape, shared by all users of the shape.
    pub fn params(&self) -> &'static Params {
        static CACHE: OnceLock<Mutex<HashMap<(usize, usize), &'static Params>>> = OnceLock::new();
        let candidate = params_for_scenario_simplepir(self.patterns, self.row_cells() * CELL_BITS);
        let key = (candidate.db_dim_1, candidate.instances);
        let mut cache = CACHE
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("PIR parameter cache");
        *cache
            .entry(key)
            .or_insert_with(|| Box::leak(Box::new(candidate)))
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

    fn to_matrices(&self, params: &'static Params) -> Result<Vec<PolyMatrixNTT<'static>>, PirError> {
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

/// The records of one claim as a PIR database, with its preprocessing.
pub struct PirDatabase {
    shape: PirShape,
    params: &'static Params,
    /// `records x record_cells` plaintext cells of the blocks.
    cells: Vec<u16>,
    records: usize,
    /// Per level above the groups, `count x UPPER_CELLS` plaintext cells:
    /// the entry of every stored node of that level, by position.
    upper: Vec<Vec<u16>>,
    /// `patterns x slots` ids; [`NO_RECORD`] for an empty slot. Slots 0 and 1
    /// hold block numbers, the slots above hold positions of upper nodes
    /// within their level.
    rows: Vec<u32>,
    offline: OfflinePrecomputedValues<'static>,
}

impl PirDatabase {
    /// Build the database of claim `registry_index` from its stored primary
    /// columns and preprocess it.
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
        Self::from_cells(layout, cells)
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

    /// [`Self::from_cells`] with the hint computed on `threads` threads,
    /// returning the time of the two preprocessing steps.
    pub fn from_cells_timed(
        layout: &PrivateLayout,
        cells: Vec<u16>,
        threads: usize,
    ) -> (Self, PreprocessTimes) {
        let shape = PirShape::of(layout);
        let records = layout.records();
        assert_eq!(cells.len(), records * shape.record_cells);
        let params = shape.params();
        let rows = layout.rows_table();
        let upper = Self::upper_from_cells(layout, &cells);
        let start = std::time::Instant::now();
        let hint = Self::hint(&shape, params, &cells, &upper, &rows, threads);
        let hint_time = start.elapsed();
        let start = std::time::Instant::now();
        let offline = factored::offline_from_hint(params, hint);
        let packing_time = start.elapsed();
        (
            Self {
                shape,
                params,
                cells,
                records,
                upper,
                rows,
                offline,
            },
            PreprocessTimes {
                hint: hint_time,
                packing: packing_time,
            },
        )
    }

    /// The plaintext values of database column `col`, one per row.
    fn fill_column(
        shape: &PirShape,
        cells: &[u16],
        upper: &[Vec<u16>],
        rows: &[u32],
        col: usize,
        values: &mut [u16],
    ) {
        values.fill(0);
        let Some((slot, offset)) = shape.slot_of_column(col) else {
            return;
        };
        let (table, width) = if slot < shape.group_slots {
            (cells, shape.record_cells)
        } else {
            (upper[slot - shape.group_slots].as_slice(), UPPER_CELLS)
        };
        for pattern in 0..shape.patterns {
            let id = rows[pattern * shape.slots + slot];
            if id != NO_RECORD {
                values[pattern] = table[id as usize * width + offset];
            }
        }
    }

    /// The SimplePIR hint, read column by column from the stored records.
    fn hint(
        shape: &PirShape,
        params: &'static Params,
        cells: &[u16],
        upper: &[Vec<u16>],
        rows: &[u32],
        threads: usize,
    ) -> Vec<u64> {
        let db_cols = factored::simplepir_db_cols(params);
        let preprocessed_query = factored::pseudorandom_query(params, SEED_0);
        let fill_column = |col: usize, values: &mut [u16]| {
            Self::fill_column(shape, cells, upper, rows, col, values)
        };
        let threads = threads.clamp(1, db_cols);
        if threads == 1 {
            return factored::hint_from_columns(params, &preprocessed_query, 0..db_cols, fill_column);
        }
        // The columns are independent: each thread takes a range of them.
        let per_thread = db_cols.div_ceil(threads);
        let ranges = (0..threads)
            .map(|k| (k * per_thread).min(db_cols)..((k + 1) * per_thread).min(db_cols))
            .filter(|range| !range.is_empty())
            .collect::<Vec<_>>();
        let preprocessed_query = &preprocessed_query;
        let fill_column = &fill_column;
        let parts = std::thread::scope(|scope| {
            let handles = ranges
                .iter()
                .cloned()
                .map(|range| {
                    scope.spawn(move || {
                        factored::hint_from_columns(params, preprocessed_query, range, fill_column)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("hint worker"))
                .collect::<Vec<_>>()
        });
        let mut hint = vec![0u64; db_cols * params.poly_len];
        for (range, part) in ranges.iter().zip(parts) {
            let width = range.len();
            for i in 0..params.poly_len {
                hint[i * db_cols + range.start..i * db_cols + range.end]
                    .copy_from_slice(&part[i * width..(i + 1) * width]);
            }
        }
        hint
    }

    /// Bring the database to the current fragment of the claim, for a log
    /// that kept its threshold and record size and did not shrink.
    ///
    /// The preprocessing is linear in the database, so only the difference
    /// is preprocessed. A record keeps its number and a slot keeps its
    /// columns, so the difference is a list of runs: consecutive patterns
    /// whose record in one slot changed in the same way. Each run adds its
    /// change, times the hint of its row range, to the columns of its slot.
    /// The response-packing tables are then derived from the patched hint.
    pub fn update(
        &mut self,
        store: &UnauthStore,
        layout: &PrivateLayout,
        registry_index: usize,
    ) -> UpdateTimes {
        self.update_records(layout, None, &mut |record, cells| {
            entries_to_cells(&record_entries(store, layout, registry_index, record), cells)
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
        assert_eq!(shape.record_cells, self.shape.record_cells, "record size changed");
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

        // The hint, moved to the new column count.
        let old_cols = factored::simplepir_db_cols(self.params);
        let mut hint = vec![0u64; db_cols * params.poly_len];
        for i in 0..params.poly_len {
            hint[i * db_cols..i * db_cols + old_cols]
                .copy_from_slice(&self.offline.hint_0[i * old_cols..(i + 1) * old_cols]);
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
        // range. The sum is taken on tiles of adjacent columns, in the two
        // residues of the ciphertext modulus, so each tile is added to
        // adjacent hint values.
        const TILE: usize = 16;
        // A residue and a reduced change are below 2^28, so this many
        // products fit in 64 bits.
        const REDUCE_EVERY: usize = 128;
        let poly_len = params.poly_len;
        let (modulus_lo, modulus_hi) = (params.moduli[0] as i64, params.moduli[1] as i64);
        let mut sum_lo = vec![0u64; poly_len * TILE];
        let mut sum_hi = vec![0u64; poly_len * TILE];
        for slot in 0..shape.slots {
            let width = shape.slot_width(slot);
            let column_base = shape.slot_offset(slot);
            let mut runs: Vec<(Option<&[u16]>, Option<&[u16]>, std::ops::Range<usize>)> = Vec::new();
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
            // The hint of each run's row range, in the two residues,
            // computed when a cell of the run first differs.
            let mut range_hints: Vec<Option<(Vec<u64>, Vec<u64>)>> = vec![None; runs.len()];
            for tile_start in (0..width).step_by(TILE) {
                let tile_width = TILE.min(width - tile_start);
                let mut touched = false;
                let mut pending = 0usize;
                for (run, (before, after, row_range)) in runs.iter().enumerate() {
                    let mut delta_lo = [0u64; TILE];
                    let mut delta_hi = [0u64; TILE];
                    let mut any = false;
                    for t in 0..tile_width {
                        let offset = tile_start + t;
                        let delta = after.map_or(0, |cells| cells[offset] as i64)
                            - before.map_or(0, |cells| cells[offset] as i64);
                        any |= delta != 0;
                        delta_lo[t] = delta.rem_euclid(modulus_lo) as u64;
                        delta_hi[t] = delta.rem_euclid(modulus_hi) as u64;
                    }
                    if !any {
                        continue;
                    }
                    if !touched {
                        sum_lo.fill(0);
                        sum_hi.fill(0);
                        touched = true;
                    }
                    let (range_lo, range_hi) = range_hints[run].get_or_insert_with(|| {
                        let values =
                            factored::hint_of_row_range(params, &preprocessed_query, row_range.clone());
                        (
                            values.iter().map(|&v| v % params.moduli[0]).collect(),
                            values.iter().map(|&v| v % params.moduli[1]).collect(),
                        )
                    });
                    for i in 0..poly_len {
                        let (lo, hi) = (range_lo[i], range_hi[i]);
                        let row_lo = &mut sum_lo[i * TILE..(i + 1) * TILE];
                        let row_hi = &mut sum_hi[i * TILE..(i + 1) * TILE];
                        for t in 0..TILE {
                            row_lo[t] += delta_lo[t] * lo;
                            row_hi[t] += delta_hi[t] * hi;
                        }
                    }
                    pending += 1;
                    if pending == REDUCE_EVERY {
                        for (lo, hi) in sum_lo.iter_mut().zip(sum_hi.iter_mut()) {
                            *lo = barrett_coeff_u64(params, *lo, 0);
                            *hi = barrett_coeff_u64(params, *hi, 1);
                        }
                        pending = 0;
                    }
                }
                if !touched {
                    continue;
                }
                for i in 0..poly_len {
                    let base = i * db_cols + column_base + tile_start;
                    for t in 0..tile_width {
                        let lo = barrett_coeff_u64(params, sum_lo[i * TILE + t], 0);
                        let hi = barrett_coeff_u64(params, sum_hi[i * TILE + t], 1);
                        let add = barrett_u64(params, params.crt_compose_2(lo, hi));
                        let value = &mut hint[base + t];
                        *value += add;
                        if *value >= params.modulus {
                            *value -= params.modulus;
                        }
                    }
                }
            }
        }

        let hint_time = start.elapsed();

        // The tables of the old hint are released before the new ones are
        // derived, so one set is held at a time.
        let start = std::time::Instant::now();
        self.offline.prepacked_lwe = Vec::new();
        self.offline.precomp = Vec::new();
        self.offline = factored::offline_from_hint(params, hint);
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
    /// those columns; the cost of one column is the cost per column of the
    /// whole hint.
    pub fn hint_of_columns(&self, cols: &[usize]) -> Vec<Vec<u64>> {
        let shape = &self.shape;
        let preprocessed_query = factored::pseudorandom_query(self.params, SEED_0);
        cols.iter()
            .map(|&col| {
                factored::hint_from_columns(
                    self.params,
                    &preprocessed_query,
                    col..col + 1,
                    |col, values| {
                        Self::fill_column(shape, &self.cells, &self.upper, &self.rows, col, values)
                    },
                )
            })
            .collect()
    }

    /// Derive the response-packing table of one response ciphertext from
    /// the stored hint and return the bytes that the state keeps of it.
    /// The tables of a database are those of its [`PirShape::outputs`]
    /// ciphertexts, each derived this way; the prepacked columns of the
    /// output are an input to the table and are released with it.
    pub fn packing_table_of_output(&self, output: usize) -> usize {
        let (_prepacked, (first, rest, _)) = factored::packing_table_of_output(
            self.params,
            &self.offline.hint_0,
            output,
            &self.offline.fake_pack_pub_params,
            &self.offline.y_constants,
        );
        first.as_slice().len() * 8 + rest.iter().map(|table| table.as_slice().len() * 8).sum::<usize>()
    }

    /// The stored hint of one database column: `poly_len` values.
    pub fn stored_hint_column(&self, col: usize) -> Vec<u64> {
        let db_cols = factored::simplepir_db_cols(self.params);
        (0..self.params.poly_len)
            .map(|i| self.offline.hint_0[i * db_cols + col])
            .collect()
    }

    /// Bytes of the preprocessing: the hint and the response-packing tables.
    pub fn preprocessed_bytes(&self) -> usize {
        let poly = |matrix: &PolyMatrixNTT<'static>| matrix.as_slice().len() * 8;
        let offline = &self.offline;
        let hint = offline.hint_0.len() * 8;
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
                    + rest.iter().map(|table| table.as_slice().len() * 8).sum::<usize>()
                    + indices
                        .iter()
                        .map(|list| list.len() * std::mem::size_of::<usize>())
                        .sum::<usize>()
            })
            .sum::<usize>();
        hint + prepacked + tables
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

    /// The SimplePIR hint of the database.
    pub fn hint_values(&self) -> &[u64] {
        &self.offline.hint_0
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
        let mut touched: Vec<u32> = Vec::new();
        let mut sum_lo = vec![0u64; shape.record_cells];
        let mut sum_hi = vec![0u64; shape.record_cells];
        // A reduced coefficient is below 2^32 and a cell below 2^16, so this
        // many products fit in 64 bits.
        const REDUCE_EVERY: usize = 1 << 15;

        for slot in 0..shape.group_slots {
            touched.clear();
            for pattern in 0..shape.patterns {
                let record = self.rows[pattern * shape.slots + slot];
                if record == NO_RECORD {
                    continue;
                }
                let g = record as usize;
                if acc_lo[g] == 0 && acc_hi[g] == 0 {
                    touched.push(record);
                }
                // Coefficients are below 2^32; reduce before the sum can
                // reach 2^64.
                acc_lo[g] += query[pattern] & LOW_MASK;
                acc_hi[g] += query[pattern] >> 32;
                if acc_lo[g] >= 1 << 62 {
                    acc_lo[g] = barrett_coeff_u64(params, acc_lo[g], 0);
                }
                if acc_hi[g] >= 1 << 62 {
                    acc_hi[g] = barrett_coeff_u64(params, acc_hi[g], 1);
                }
            }
            sum_lo.fill(0);
            sum_hi.fill(0);
            let mut pending = 0usize;
            // A record whose coefficients sum to zero is listed twice at
            // most; its second visit adds zero.
            for &record in &touched {
                let g = record as usize;
                let a_lo = barrett_coeff_u64(params, acc_lo[g], 0);
                let a_hi = barrett_coeff_u64(params, acc_hi[g], 1);
                acc_lo[g] = 0;
                acc_hi[g] = 0;
                add_scaled_cells(&mut sum_lo, &mut sum_hi, a_lo, a_hi, self.record_cells(g));
                pending += 1;
                if pending == REDUCE_EVERY {
                    for (lo, hi) in sum_lo.iter_mut().zip(sum_hi.iter_mut()) {
                        *lo = barrett_coeff_u64(params, *lo, 0);
                        *hi = barrett_coeff_u64(params, *hi, 1);
                    }
                    pending = 0;
                }
            }
            let base = shape.slot_offset(slot);
            for offset in 0..shape.record_cells {
                let lo = barrett_coeff_u64(params, sum_lo[offset], 0);
                let hi = barrett_coeff_u64(params, sum_hi[offset], 1);
                out[base + offset] = barrett_u64(params, params.crt_compose_2(lo, hi));
            }
        }

        // Above the groups, each stored node is multiplied once by the sum
        // of the coefficients of the patterns below its sibling, a
        // contiguous range.
        let mut upper_lo = [0u64; UPPER_CELLS];
        let mut upper_hi = [0u64; UPPER_CELLS];
        for level in 0..shape.upper_slots() {
            upper_lo.fill(0);
            upper_hi.fill(0);
            let mut pending = 0usize;
            let count = self.upper[level].len() / UPPER_CELLS;
            for pos in 0..count {
                let start = ((pos ^ 1) << level).min(shape.patterns);
                let end = (((pos ^ 1) + 1) << level).min(shape.patterns);
                let mut lo = 0u64;
                let mut hi = 0u64;
                for &coefficient in &query[start..end] {
                    lo += coefficient & LOW_MASK;
                    hi += coefficient >> 32;
                    if lo >= 1 << 62 {
                        lo = barrett_coeff_u64(params, lo, 0);
                    }
                    if hi >= 1 << 62 {
                        hi = barrett_coeff_u64(params, hi, 1);
                    }
                }
                let a_lo = barrett_coeff_u64(params, lo, 0);
                let a_hi = barrett_coeff_u64(params, hi, 1);
                if a_lo == 0 && a_hi == 0 {
                    continue;
                }
                for ((lo, hi), &cell) in upper_lo
                    .iter_mut()
                    .zip(upper_hi.iter_mut())
                    .zip(self.upper_cells(level, pos))
                {
                    *lo += a_lo * cell as u64;
                    *hi += a_hi * cell as u64;
                }
                pending += 1;
                if pending == REDUCE_EVERY {
                    for (lo, hi) in upper_lo.iter_mut().zip(upper_hi.iter_mut()) {
                        *lo = barrett_coeff_u64(params, *lo, 0);
                        *hi = barrett_coeff_u64(params, *hi, 1);
                    }
                    pending = 0;
                }
            }
            let base = shape.slot_offset(shape.group_slots + level);
            for offset in 0..UPPER_CELLS {
                let lo = barrett_coeff_u64(params, upper_lo[offset], 0);
                let hi = barrett_coeff_u64(params, upper_hi[offset], 1);
                out[base + offset] = barrett_u64(params, params.crt_compose_2(lo, hi));
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

/// `sum_lo[k] += a_lo * cells[k]` and `sum_hi[k] += a_hi * cells[k]`: the
/// product of one stored item with a coefficient in its two residues. The
/// coefficients are below `2^32`, so the products fit the lanes of the AVX2
/// kernel, which is used when the host has it.
fn add_scaled_cells(sum_lo: &mut [u64], sum_hi: &mut [u64], a_lo: u64, a_hi: u64, cells: &[u16]) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static AVX2: OnceLock<bool> = OnceLock::new();
        if *AVX2.get_or_init(|| is_x86_feature_detected!("avx2")) {
            // SAFETY: the host supports AVX2.
            unsafe { add_scaled_cells_avx2(sum_lo, sum_hi, a_lo, a_hi, cells) };
            return;
        }
    }
    for ((lo, hi), &cell) in sum_lo.iter_mut().zip(sum_hi.iter_mut()).zip(cells) {
        *lo += a_lo * cell as u64;
        *hi += a_hi * cell as u64;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_cells_avx2(
    sum_lo: &mut [u64],
    sum_hi: &mut [u64],
    a_lo: u64,
    a_hi: u64,
    cells: &[u16],
) {
    use std::arch::x86_64::*;
    let n = cells.len();
    let lo_ptr = sum_lo.as_mut_ptr();
    let hi_ptr = sum_hi.as_mut_ptr();
    let cell_ptr = cells.as_ptr();
    let mut k = 0;
    // SAFETY: every access stays within the slices, four lanes at a time.
    unsafe {
        let a_lo_v = _mm256_set1_epi64x(a_lo as i64);
        let a_hi_v = _mm256_set1_epi64x(a_hi as i64);
        while k + 4 <= n {
            let four = _mm_loadl_epi64(cell_ptr.add(k) as *const _);
            let c = _mm256_cvtepu16_epi64(four);
            let lo = _mm256_loadu_si256(lo_ptr.add(k) as *const _);
            let hi = _mm256_loadu_si256(hi_ptr.add(k) as *const _);
            _mm256_storeu_si256(
                lo_ptr.add(k) as *mut _,
                _mm256_add_epi64(lo, _mm256_mul_epu32(a_lo_v, c)),
            );
            _mm256_storeu_si256(
                hi_ptr.add(k) as *mut _,
                _mm256_add_epi64(hi, _mm256_mul_epu32(a_hi_v, c)),
            );
            k += 4;
        }
    }
    while k < n {
        sum_lo[k] += a_lo * cells[k] as u64;
        sum_hi[k] += a_hi * cells[k] as u64;
        k += 1;
    }
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
            .map(|slot| {
                let offset = self.shape.slot_offset(slot);
                cells[offset..offset + self.shape.slot_width(slot)].to_vec()
            })
            .collect())
    }
}

/// The table of complete openings: every row of the PIR database stored in
/// full and served by YPIR's own server.
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
    /// Write out every row of `database` and preprocess the table.
    pub fn from_database(database: &PirDatabase) -> Self {
        let params = database.params;
        let shape = database.shape;
        let db_rows = factored::simplepir_db_rows(params);
        let db_cols = factored::simplepir_db_cols(params);
        // `YServer::new` reads its input row by row.
        let table = (0..db_rows).flat_map(|row| {
            (0..db_cols).map(move |col| {
                let Some((slot, offset)) = shape.slot_of_column(col) else {
                    return 0u16;
                };
                if row >= shape.patterns {
                    return 0u16;
                }
                match database.rows[row * shape.slots + slot] {
                    NO_RECORD => 0,
                    id => database.slot_cells(slot, id as usize)[offset],
                }
            })
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
        let aligned = self.aligned_query(query)?;
        Ok(self.server.answer_query(aligned.as_slice()).as_slice().to_vec())
    }

    /// Answer one query with YPIR's server.
    pub fn answer(&self, query: &[u64], pack_keys: &PackKeys) -> Result<Vec<Vec<u8>>, PirError> {
        let pack_keys = pack_keys.to_matrices(self.params)?;
        let aligned = self.aligned_query(query)?;
        Ok(self.server.perform_online_computation_simplepir(
            aligned.as_slice(),
            &self.offline,
            &[pack_keys.as_slice()],
            None,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
