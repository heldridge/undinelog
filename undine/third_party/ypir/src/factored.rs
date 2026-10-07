//! SimplePIR-variant building blocks for a database given by its columns.
//!
//! `YServer` stores the database as a matrix. The functions here run the same
//! computations on a database that the caller describes column by column, so
//! a server whose rows repeat stored records can answer without building the
//! matrix. Each function follows the corresponding part of `YServer` and of
//! `run_simple_ypir_on_params` step for step, and the test at the end checks
//! that the results equal `YServer`'s byte for byte.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::{arith::*, client::*, number_theory::*, params::*, poly::*};

use crate::client::*;
use crate::modulus_switch::ModulusSwitch;
use crate::packing::*;
use crate::params::*;
use crate::scheme::*;
use crate::server::*;
use crate::transpose::*;
use crate::util::*;

/// Rows of the database: `2^(db_dim_1 + poly_len_log2)`.
pub fn simplepir_db_rows(params: &Params) -> usize {
    1 << (params.db_dim_1 + params.poly_len_log2)
}

/// Columns of the database: `instances * poly_len`.
pub fn simplepir_db_cols(params: &Params) -> usize {
    params.instances * params.poly_len
}

/// The public pseudorandom query of seed `public_seed_idx`, in the form the
/// hint computation multiplies with the database
/// (`YServer::generate_pseudorandom_query`).
pub fn pseudorandom_query<'a>(params: &'a Params, public_seed_idx: u8) -> Vec<PolyMatrixNTT<'a>> {
    let mut client = Client::init(params);
    client.generate_secret_keys();
    let y_client = YClient::new(&mut client, params);
    let query = y_client.generate_query_impl(public_seed_idx, params.db_dim_1, true, 0);
    let mut preprocessed_query = Vec::with_capacity(query.len());
    for query_raw in query.iter().map(|x| x.submatrix(0, 0, 1, 1)) {
        let query_raw_transformed = negacyclic_perm(query_raw.get_poly(0, 0), 0, params.modulus);
        let mut query_transformed_pol = PolyMatrixRaw::zero(params, 1, 1);
        query_transformed_pol
            .as_mut_slice()
            .copy_from_slice(&query_raw_transformed);
        preprocessed_query.push(query_transformed_pol.ntt());
    }
    preprocessed_query
}

/// The SimplePIR hint of the columns `cols` of a database
/// (`YServer::answer_hint_ring` with `SEED_0`).
///
/// `fill_column(col, values)` writes the `db_rows` plaintext values of column
/// `col`. The result holds `poly_len` values per column, laid out as
/// `[i * cols.len() + (col - cols.start)]`.
pub fn hint_from_columns<'a, F>(
    params: &'a Params,
    preprocessed_query: &[PolyMatrixNTT<'a>],
    cols: std::ops::Range<usize>,
    mut fill_column: F,
) -> Vec<u64>
where
    F: FnMut(usize, &mut [u16]),
{
    let db_rows_poly = 1 << params.db_dim_1;
    let db_rows = simplepir_db_rows(params);
    assert_eq!(preprocessed_query.len(), db_rows_poly);

    let mut result = Vec::with_capacity(cols.len() * params.poly_len);
    let mut column = vec![0u16; db_rows];
    let mut prod = PolyMatrixNTT::zero(params, 1, 1);
    let mut db_elem_poly = PolyMatrixRaw::zero(params, 1, 1);
    let mut db_elem_ntt = PolyMatrixNTT::zero(params, 1, 1);

    for col in cols.clone() {
        fill_column(col, &mut column);
        let mut sum = PolyMatrixNTT::zero(params, 1, 1);

        for row in 0..db_rows_poly {
            let block = &column[row * params.poly_len..(row + 1) * params.poly_len];
            if block.iter().all(|&x| x == 0) {
                // The product with a zero block is zero.
                prod.as_mut_slice().fill(0);
            } else {
                for z in 0..params.poly_len {
                    db_elem_poly.data[z] = block[z] as u64;
                }
                to_ntt(&mut db_elem_ntt, &db_elem_poly);
                multiply(&mut prod, &preprocessed_query[row], &db_elem_ntt);
            }

            if row == db_rows_poly - 1 {
                add_into(&mut sum, &prod);
            } else {
                add_into_no_reduce(&mut sum, &prod);
            }
        }

        let sum_raw = sum.raw();
        result.extend(sum_raw.as_slice());
    }

    transpose_generic(&result, cols.len(), params.poly_len)
}

/// [`hint_from_columns`] in column-major order: the hint of column `col`
/// of `cols` is `result[(col - cols.start) * poly_len..][..poly_len]`.
pub fn hint_columns_from_columns<'a, F>(
    params: &'a Params,
    preprocessed_query: &[PolyMatrixNTT<'a>],
    cols: std::ops::Range<usize>,
    fill_column: F,
) -> Vec<u64>
where
    F: FnMut(usize, &mut [u16]),
{
    let width = cols.len();
    let row_major = hint_from_columns(params, preprocessed_query, cols, fill_column);
    transpose_generic(&row_major, params.poly_len, width)
}

/// The hint of a column that is `1` on the rows `rows` and `0` elsewhere:
/// `poly_len` values modulo the ciphertext modulus.
///
/// The hint is linear in the database, so a column that changes by `delta` on
/// `rows` changes its hint by `delta` times this vector.
pub fn hint_of_row_range<'a>(
    params: &'a Params,
    preprocessed_query: &[PolyMatrixNTT<'a>],
    rows: std::ops::Range<usize>,
) -> Vec<u64> {
    assert!(rows.end <= simplepir_db_rows(params));
    let mut sum = PolyMatrixNTT::zero(params, 1, 1);
    if rows.is_empty() {
        return sum.raw().as_slice().to_vec();
    }
    let mut prod = PolyMatrixNTT::zero(params, 1, 1);
    let mut db_elem_poly = PolyMatrixRaw::zero(params, 1, 1);
    let mut db_elem_ntt = PolyMatrixNTT::zero(params, 1, 1);
    let first_block = rows.start / params.poly_len;
    let last_block = (rows.end - 1) / params.poly_len;
    for block in first_block..=last_block {
        let block_start = block * params.poly_len;
        for z in 0..params.poly_len {
            db_elem_poly.data[z] = rows.contains(&(block_start + z)) as u64;
        }
        to_ntt(&mut db_elem_ntt, &db_elem_poly);
        multiply(&mut prod, &preprocessed_query[block], &db_elem_ntt);
        add_into(&mut sum, &prod);
    }
    sum.raw().as_slice().to_vec()
}

/// Add `delta * range_hint` to column `col` of a hint laid out as
/// `[i * db_cols + col]`, modulo the ciphertext modulus. `delta` is a signed
/// change of a plaintext value.
pub fn add_scaled_to_hint_column(
    params: &Params,
    hint: &mut [u64],
    db_cols: usize,
    col: usize,
    range_hint: &[u64],
    delta: i64,
) {
    assert_eq!(range_hint.len(), params.poly_len);
    assert_eq!(hint.len(), db_cols * params.poly_len);
    let modulus = params.modulus as u128;
    let scale = delta.rem_euclid(params.modulus as i64) as u128;
    for (i, &value) in range_hint.iter().enumerate() {
        let slot = &mut hint[i * db_cols + col];
        *slot = ((*slot as u128 + scale * value as u128) % modulus) as u64;
    }
}

/// Everything the online phase needs, from the hint of all columns
/// (`YServer::perform_offline_precomputation_simplepir` after its hint).
pub fn offline_from_hint<'a>(params: &'a Params, hint_0: Vec<u64>) -> OfflinePrecomputedValues<'a> {
    let db_cols = simplepir_db_cols(params);
    let num_rlwe_outputs = db_cols / params.poly_len;
    assert_eq!(hint_0.len(), db_cols * params.poly_len);

    let y_constants = generate_y_constants(params);

    let fake_pack_pub_params = generate_fake_pack_pub_params(params);

    // The prepacked columns of one output are an input to its table only:
    // they are derived per output and released, so the state holds the
    // hint and the tables.
    let mut precomp: Precomp = Vec::with_capacity(num_rlwe_outputs);
    for output in 0..num_rlwe_outputs {
        let (_, table) = packing_table_of_output(
            params,
            &hint_0,
            output,
            &fake_pack_pub_params,
            &y_constants,
        );
        precomp.push(table);
    }

    OfflinePrecomputedValues {
        hint_0,
        hint_1: vec![],
        pseudorandom_query_1: vec![],
        y_constants,
        smaller_server: None,
        prepacked_lwe: Vec::new(),
        fake_pack_pub_params,
        precomp,
    }
}

/// [`offline_from_hint`] from a column-major hint, `hint_cols[col * poly_len + j]`,
/// which the caller keeps: the returned values hold the tables and no hint.
pub fn offline_from_hint_columns<'a>(
    params: &'a Params,
    hint_cols: &[u64],
) -> OfflinePrecomputedValues<'a> {
    offline_from_hint_columns_on(params, hint_cols, 1)
}

/// [`offline_from_hint_columns`] with the tables of the outputs derived on
/// `threads` threads; each thread holds the table of one output at a time.
pub fn offline_from_hint_columns_on<'a>(
    params: &'a Params,
    hint_cols: &[u64],
    threads: usize,
) -> OfflinePrecomputedValues<'a> {
    let db_cols = simplepir_db_cols(params);
    let num_rlwe_outputs = db_cols / params.poly_len;
    assert_eq!(hint_cols.len(), db_cols * params.poly_len);
    let y_constants = generate_y_constants(params);
    let fake_pack_pub_params = generate_fake_pack_pub_params(params);
    let table_of = |output: usize| {
        packing_table_of_columns(params, hint_cols, output, &fake_pack_pub_params, &y_constants).1
    };
    let threads = threads.clamp(1, num_rlwe_outputs.max(1));
    let precomp: Precomp = if threads == 1 {
        (0..num_rlwe_outputs).map(table_of).collect()
    } else {
        // The outputs are independent; each thread takes every `threads`-th.
        let table_of = &table_of;
        std::thread::scope(|scope| {
            let handles = (0..threads)
                .map(|k| {
                    scope.spawn(move || {
                        (k..num_rlwe_outputs).step_by(threads).map(table_of).collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>();
            let mut parts = handles
                .into_iter()
                .map(|h| h.join().expect("table worker").into_iter())
                .collect::<Vec<_>>();
            (0..num_rlwe_outputs)
                .map(|output| parts[output % threads].next().expect("one table per output"))
                .collect()
        })
    };
    OfflinePrecomputedValues {
        hint_0: Vec::new(),
        hint_1: vec![],
        pseudorandom_query_1: vec![],
        y_constants,
        smaller_server: None,
        prepacked_lwe: Vec::new(),
        fake_pack_pub_params,
        precomp,
    }
}

/// The response-packing tables of RLWE output `output`, from the hint of all
/// columns: the entries `prepacked_lwe[output]` and `precomp[output]` of
/// [`offline_from_hint`], with its `fake_pack_pub_params` and `y_constants`.
pub fn packing_table_of_output<'a>(
    params: &'a Params,
    hint_0: &[u64],
    output: usize,
    fake_pack_pub_params: &[PolyMatrixNTT<'a>],
    y_constants: &(Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>),
) -> (
    Vec<PolyMatrixNTT<'a>>,
    (PolyMatrixNTT<'a>, Vec<crate::packing::Condensed>, Vec<Vec<usize>>),
) {
    let db_cols = simplepir_db_cols(params);
    assert_eq!(hint_0.len(), db_cols * params.poly_len);
    assert!(output < db_cols / params.poly_len);

    // The LWE ciphertexts of this output: their hint rows and a zero row.
    let mut lwe_cts = Vec::with_capacity((params.poly_len + 1) * params.poly_len);
    for j in 0..params.poly_len {
        lwe_cts.extend(&hint_0[j * db_cols + output * params.poly_len..][..params.poly_len]);
    }
    packing_table_of_block(params, lwe_cts, fake_pack_pub_params, y_constants)
}

/// [`packing_table_of_output`] from a column-major hint,
/// `hint_cols[col * poly_len + j]`.
pub fn packing_table_of_columns<'a>(
    params: &'a Params,
    hint_cols: &[u64],
    output: usize,
    fake_pack_pub_params: &[PolyMatrixNTT<'a>],
    y_constants: &(Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>),
) -> (
    Vec<PolyMatrixNTT<'a>>,
    (PolyMatrixNTT<'a>, Vec<crate::packing::Condensed>, Vec<Vec<usize>>),
) {
    let db_cols = simplepir_db_cols(params);
    let n = params.poly_len;
    assert_eq!(hint_cols.len(), db_cols * n);
    assert!(output < db_cols / n);
    let mut lwe_cts = vec![0u64; n * n];
    for c in 0..n {
        let column = &hint_cols[(output * n + c) * n..][..n];
        for j in 0..n {
            lwe_cts[j * n + c] = column[j];
        }
    }
    packing_table_of_block(params, lwe_cts, fake_pack_pub_params, y_constants)
}

/// The packing table of one output from the row-major block of its hint,
/// `block[j * poly_len + c]` for the `poly_len` columns of the output.
pub fn packing_table_of_block<'a>(
    params: &'a Params,
    mut lwe_cts: Vec<u64>,
    fake_pack_pub_params: &[PolyMatrixNTT<'a>],
    y_constants: &(Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>),
) -> (
    Vec<PolyMatrixNTT<'a>>,
    (PolyMatrixNTT<'a>, Vec<crate::packing::Condensed>, Vec<Vec<usize>>),
) {
    assert_eq!(lwe_cts.len(), params.poly_len * params.poly_len);
    lwe_cts.resize((params.poly_len + 1) * params.poly_len, 0);
    let prepacked = prep_pack_lwes(params, &lwe_cts, params.poly_len);
    let table = precompute_pack(
        params,
        params.poly_len_log2,
        &prepacked,
        fake_pack_pub_params,
        y_constants,
    );
    (prepacked, table)
}

/// The response, from the result of the first pass over the database
/// (`YServer::perform_online_computation_simplepir` after its dot product).
///
/// `first_pass[col]` is the inner product of the packed query with column
/// `col`, reduced modulo the ciphertext modulus.
pub fn response_from_first_pass<'a>(
    params: &'a Params,
    offline_vals: &OfflinePrecomputedValues<'a>,
    first_pass: &[u64],
    pack_pub_params_row_1s: &[PolyMatrixNTT<'a>],
) -> Vec<Vec<u8>> {
    let db_cols = simplepir_db_cols(params);
    assert_eq!(first_pass.len(), db_cols);
    let rlwe_q_prime_1 = params.get_q_prime_1();
    let rlwe_q_prime_2 = params.get_q_prime_2();
    let num_rlwe_outputs = db_cols / params.poly_len;

    let packed = pack_many_lwes(
        params,
        &offline_vals.prepacked_lwe,
        &offline_vals.precomp,
        first_pass,
        num_rlwe_outputs,
        pack_pub_params_row_1s,
        &offline_vals.y_constants,
    );

    let mut packed_mod_switched = Vec::with_capacity(packed.len());
    for ct in packed.iter() {
        let res = ct.raw();
        packed_mod_switched.push(res.switch(rlwe_q_prime_1, rlwe_q_prime_2));
    }
    packed_mod_switched
}

/// The reply computed without packing tables: every reply ciphertext is
/// packed online from the hint of its columns and the first pass, with the
/// public half of the packing keys regenerated from the public seed.
pub fn response_from_first_pass_online<'a>(
    params: &'a Params,
    hint_0: &[u64],
    first_pass: &[u64],
    pack_pub_params_row_1s: &[PolyMatrixNTT<'a>],
) -> Vec<Vec<u8>> {
    let db_cols = simplepir_db_cols(params);
    assert_eq!(first_pass.len(), db_cols);
    let rlwe_q_prime_1 = params.get_q_prime_1();
    let rlwe_q_prime_2 = params.get_q_prime_2();
    let num_rlwe_outputs = db_cols / params.poly_len;
    let y_constants = generate_y_constants(params);
    let mut pack_pub_params = generate_fake_pack_pub_params(params);
    for i in 0..pack_pub_params.len() {
        let uncondensed = uncondense_matrix(params, &pack_pub_params_row_1s[i]);
        pack_pub_params[i].copy_into(&uncondensed, 1, 0);
    }
    let mut out = Vec::with_capacity(num_rlwe_outputs);
    let mut lwe_cts = vec![0u64; (params.poly_len + 1) * params.poly_len];
    for output in 0..num_rlwe_outputs {
        for j in 0..params.poly_len {
            lwe_cts[j * params.poly_len..(j + 1) * params.poly_len]
                .copy_from_slice(&hint_0[j * db_cols + output * params.poly_len..][..params.poly_len]);
        }
        lwe_cts[params.poly_len * params.poly_len..].fill(0);
        let prepacked = prep_pack_lwes(params, &lwe_cts, params.poly_len);
        let packed = pack_lwes(
            params,
            &first_pass[output * params.poly_len..(output + 1) * params.poly_len],
            &prepacked,
            &[],
            params.poly_len,
            &pack_pub_params,
            &y_constants,
        );
        out.push(packed.raw().switch(rlwe_q_prime_1, rlwe_q_prime_2));
    }
    out
}

/// The packing keys a client uploads with its query
/// (`run_simple_ypir_on_params`, query generation).
pub fn client_pack_pub_params<'a>(params: &'a Params, client: &Client<'a>) -> Vec<PolyMatrixNTT<'a>> {
    let sk_reg = client.get_sk_reg();
    let pack_pub_params = raw_generate_expansion_params(
        params,
        sk_reg,
        params.poly_len_log2,
        params.t_exp_left,
        &mut ChaCha20Rng::from_entropy(),
        &mut ChaCha20Rng::from_seed(STATIC_SEED_2),
    );
    let mut pack_pub_params_row_1s = pack_pub_params.to_vec();
    for i in 0..pack_pub_params.len() {
        pack_pub_params_row_1s[i] = pack_pub_params[i].submatrix(1, 0, 1, pack_pub_params[i].cols);
        pack_pub_params_row_1s[i] = condense_matrix(params, &pack_pub_params_row_1s[i]);
    }
    pack_pub_params_row_1s
}

/// The packed query for row `index_row`
/// (`YClient::generate_query` with `SEED_0` and packing, then `pack_query`).
pub fn client_query<'a>(params: &'a Params, client: &Client<'a>, index_row: usize) -> AlignedMemory64 {
    let db_rows = simplepir_db_rows(params);
    assert!(index_row < db_rows);
    let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));
    let scale_k = params.modulus / params.pt_modulus;
    let factor = invert_uint_mod(params.poly_len as u64, params.modulus).unwrap();

    let mut lwes = Vec::with_capacity(db_rows);
    for i in 0..(1 << params.db_dim_1) {
        let mut scalar = PolyMatrixRaw::zero(params, 1, 1);
        if i == index_row / params.poly_len {
            scalar.data[index_row % params.poly_len] = scale_k;
        }
        scalar = scalar_multiply_alloc(
            &PolyMatrixRaw::single_value(params, factor).ntt(),
            &scalar.ntt(),
        )
        .raw();
        let ct = client.encrypt_matrix_scaled_reg(
            &scalar.ntt(),
            &mut ChaCha20Rng::from_entropy(),
            &mut rng_pub,
            factor,
        );
        lwes.extend(rlwe_to_lwe_last_row(params, &ct.raw()));
    }
    assert_eq!(lwes.len(), db_rows);
    pack_query(params, &lwes)
}

/// The plaintext values of the selected row: one per database column
/// (`run_simple_ypir_on_params`, decoding).
pub fn client_decode<'a>(params: &'a Params, client: &Client<'a>, response: &[Vec<u8>]) -> Vec<u64> {
    let rlwe_q_prime_1 = params.get_q_prime_1();
    let rlwe_q_prime_2 = params.get_q_prime_2();
    let mut out = Vec::with_capacity(response.len() * params.poly_len);
    for ct_bytes in response {
        let ct = PolyMatrixRaw::recover(params, rlwe_q_prime_1, rlwe_q_prime_2, ct_bytes);
        let dec_result = client.decrypt_matrix_reg(&ct.ntt()).raw();
        for z in 0..params.poly_len {
            out.push(rescale(dec_result.data[z], params.modulus, params.pt_modulus));
        }
    }
    out
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::bits::*;

    /// The column-based path gives `YServer`'s hint and response byte for
    /// byte, and the client decodes the selected row.
    #[test]
    fn column_based_path_equals_yserver() {
        // 4096 rows, three RLWE outputs per row.
        let params = params_for_scenario_simplepir(4096, 3 * 2048 * 14);
        let db_rows = simplepir_db_rows(&params);
        let db_cols = simplepir_db_cols(&params);
        assert_eq!((db_rows, db_cols), (4096, 3 * 2048));

        // A database with some all-zero blocks and columns.
        let value = |row: usize, col: usize| -> u16 {
            if col % 7 == 3 || (row / 2048 == 1 && col % 5 == 0) {
                0
            } else {
                ((row as u64 * 2_654_435_761 + col as u64 * 40_503 + 17) % params.pt_modulus) as u16
            }
        };
        // `YServer::new` reads its input row by row.
        let row_major = (0..db_rows).flat_map(|row| (0..db_cols).map(move |col| value(row, col)));
        let y_server = YServer::<u16>::new(&params, row_major, true, false, true);
        let offline = y_server.perform_offline_precomputation_simplepir(None);

        let preprocessed_query = pseudorandom_query(&params, SEED_0);
        let hint = hint_from_columns(&params, &preprocessed_query, 0..db_cols, |col, values| {
            for (row, out) in values.iter_mut().enumerate() {
                *out = value(row, col);
            }
        });
        assert!(hint == offline.hint_0, "hint differs from YServer's");

        // Columns computed in two ranges and joined give the same hint.
        let split = 1000;
        let left = hint_from_columns(&params, &preprocessed_query, 0..split, |col, values| {
            for (row, out) in values.iter_mut().enumerate() {
                *out = value(row, col);
            }
        });
        let right = hint_from_columns(&params, &preprocessed_query, split..db_cols, |col, values| {
            for (row, out) in values.iter_mut().enumerate() {
                *out = value(row, col);
            }
        });
        for i in 0..params.poly_len {
            assert!(left[i * split..(i + 1) * split] == hint[i * db_cols..i * db_cols + split]);
            assert!(
                right[i * (db_cols - split)..(i + 1) * (db_cols - split)]
                    == hint[i * db_cols + split..(i + 1) * db_cols]
            );
        }

        // The hint is linear: changing a column on a range of rows changes
        // its hint by the change times the hint of that range.
        let changed_col = 4;
        let changed_rows = 1500..2600;
        let delta: i64 = 5;
        let range_hint = hint_of_row_range(&params, &preprocessed_query, changed_rows.clone());
        let changed = hint_from_columns(&params, &preprocessed_query, 0..db_cols, |col, values| {
            for (row, out) in values.iter_mut().enumerate() {
                *out = value(row, col);
                if col == changed_col && changed_rows.contains(&row) {
                    *out += delta as u16;
                }
            }
        });
        assert!(changed != hint);
        let mut patched = hint.clone();
        add_scaled_to_hint_column(&params, &mut patched, db_cols, changed_col, &range_hint, delta);
        assert!(patched == changed, "patched hint differs from a recomputed one");
        // A negative change takes the hint back.
        add_scaled_to_hint_column(&params, &mut patched, db_cols, changed_col, &range_hint, -delta);
        assert!(patched == hint, "the reverse patch does not restore the hint");

        // A database with more rows keeps the public query of the first rows.
        let taller = params_for_scenario_simplepir(8192, 3 * 2048 * 14);
        let taller_query = pseudorandom_query(&taller, SEED_0);
        assert_eq!(taller_query.len(), 2 * preprocessed_query.len());
        for (short, tall) in preprocessed_query.iter().zip(taller_query.iter()) {
            assert!(short.as_slice() == tall.as_slice());
        }

        let column_offline = offline_from_hint(&params, hint);
        let hint_for_online = column_offline.hint_0.clone();
        for output in 0..db_cols / params.poly_len {
            let (prepacked, (first, rest, indices)) = packing_table_of_output(
                &params,
                &column_offline.hint_0,
                output,
                &column_offline.fake_pack_pub_params,
                &column_offline.y_constants,
            );
            assert_eq!(prepacked.len(), params.poly_len);
            let expected = &column_offline.precomp[output];
            assert!(first.as_slice() == expected.0.as_slice());
            assert!(
                rest.len() == expected.1.len()
                    && rest.iter().zip(&expected.1).all(|(a, b)| a.as_slice() == b.as_slice())
            );
            assert!(indices == expected.2, "packing table of output {output}");
        }

        let mut client = Client::init(&params);
        client.generate_secret_keys();
        let pack_pub_params = client_pack_pub_params(&params, &client);
        for target_row in [0usize, 1, 2047, 2048, 4095] {
            let query = client_query(&params, &client, target_row);
            let expected = y_server.perform_online_computation_simplepir(
                query.as_slice(),
                &offline,
                &[pack_pub_params.as_slice()],
                None,
            );
            let first_pass = y_server.answer_query(query.as_slice());
            let response = response_from_first_pass(
                &params,
                &column_offline,
                first_pass.as_slice(),
                &pack_pub_params,
            );
            assert!(response == expected, "response differs from YServer's");
            let started = std::time::Instant::now();
            let online = response_from_first_pass_online(
                &params,
                &hint_for_online,
                first_pass.as_slice(),
                &pack_pub_params,
            );
            let online_time = started.elapsed();
            let started = std::time::Instant::now();
            let _ = response_from_first_pass(&params, &column_offline, first_pass.as_slice(), &pack_pub_params);
            let table_time = started.elapsed();
            eprintln!(
                "packing of {} ciphertexts: online {:?}, with tables {:?}",
                online.len(),
                online_time,
                table_time
            );

            let decoded = client_decode(&params, &client, &response);
            let row = (0..db_cols).map(|col| value(target_row, col) as u64).collect::<Vec<_>>();
            assert!(decoded == row, "decoded row {target_row} is wrong");
            let decoded_online = client_decode(&params, &client, &online);
            assert!(decoded_online == row, "online packing decodes row {target_row} wrong");
        }
    }
}
