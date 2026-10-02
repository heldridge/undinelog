use ark_ff::Field;
use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::unauth::{
    ClaimBundle, RecoveredClaimSegmentState, RowKey, SourceHalfState, UnauthStore,
    aligned_block_indices, block_grid_indices, build_kappa_double_params, build_kappa_halve_params,
    logical_offset_to_grid_index,
};
use paper_a_optimized::{
    CodedArrayParams, Fp, build_coded_params, build_shard_registry_deterministic, encode_value,
    fp_one, fp_pow, fp_zero,
};

/// State of one claim for one segment, computed from the segment polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectClaim {
    column_identifier: Fp,
    row_label: Fp,
    column: Vec<Fp>,
    row: Vec<Fp>,
}

fn build_params(kappa: usize, shard_count: usize) -> CodedArrayParams {
    let cfg = CodedArrayConfig {
        kappa,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 23,
    };
    build_coded_params(&cfg)
}

/// Coefficients of the polynomial of one segment:
/// `P(Z, W) = sum_{a, b} coeffs[a * kappa + b] Z^a W^b`.
fn sample_segment_coeffs(n_grid: usize, seg: usize) -> Vec<Fp> {
    (0..n_grid)
        .map(|idx| Fp::from((1_000 * seg as u64) + (7 * idx as u64) + ((idx % 5) as u64) + 3))
        .collect()
}

fn eval_poly(coeffs: &[Fp], x: Fp) -> Fp {
    let mut acc = fp_zero();
    for &coeff in coeffs.iter().rev() {
        acc *= x;
        acc += coeff;
    }
    acc
}

fn eval_segment_poly(coeffs: &[Fp], kappa: usize, z: Fp, w: Fp) -> Fp {
    assert_eq!(coeffs.len(), kappa * kappa);
    let mut acc = fp_zero();
    for z_coeffs in coeffs.chunks_exact(kappa).rev() {
        acc *= z;
        acc += eval_poly(z_coeffs, w);
    }
    acc
}

fn z_axis_points(params: &CodedArrayParams, seg: usize) -> Vec<Fp> {
    let rho_z = params.segment_axes(seg).rho_z;
    params.axis.iter().map(|&point| rho_z * point).collect()
}

fn w_axis_points(params: &CodedArrayParams, seg: usize) -> Vec<Fp> {
    let rho_w = params.segment_axes(seg).rho_w;
    params.axis.iter().map(|&point| rho_w * point).collect()
}

/// Cells of one segment in grid order: the values of its polynomial on the
/// product of its two axes.
fn direct_cells(params: &CodedArrayParams, seg: usize, coeffs: &[Fp]) -> Vec<Fp> {
    let w_points = w_axis_points(params, seg);
    let mut cells = Vec::with_capacity(params.n_grid);
    for z in z_axis_points(params, seg) {
        for &w in &w_points {
            cells.push(eval_segment_poly(coeffs, params.need, z, w));
        }
    }
    cells
}

fn build_direct_claim(
    params: &CodedArrayParams,
    seg: usize,
    coeffs: &[Fp],
    registry_index: usize,
) -> DirectClaim {
    let column_identifier = params.shard_registry[registry_index].z;
    let row_label = params.row_labels[registry_index];
    let column = w_axis_points(params, seg)
        .into_iter()
        .map(|eta| eval_segment_poly(coeffs, params.need, column_identifier, eta))
        .collect();
    let row = z_axis_points(params, seg)
        .into_iter()
        .map(|z| eval_segment_poly(coeffs, params.need, z, row_label))
        .collect();
    DirectClaim {
        column_identifier,
        row_label,
        column,
        row,
    }
}

/// Selectors of the two halves of the axis `rho * H_{2 kappa}` at `x`:
/// `(1 + (x / rho)^kappa) / 2` and `(1 - (x / rho)^kappa) / 2`.
fn axis_selectors(kappa: usize, rho: Fp, x: Fp) -> [Fp; 2] {
    let half_inv = Fp::from(2u64).inverse().expect("2 invertible");
    let power = fp_pow(
        x * rho.inverse().expect("axis scale invertible"),
        kappa as u64,
    );
    [(fp_one() + power) * half_inv, (fp_one() - power) * half_inv]
}

/// Evaluate the polynomial of stitched segment `parent` at `(z, w)`:
/// `sum_{u, v} e_u(z) f_v(w) P_{uv}(z, w)`, where `P_{uv}` is the polynomial
/// of source segment `4 * parent + 2u + v`.
fn eval_stitched_poly(
    source_params: &CodedArrayParams,
    target_params: &CodedArrayParams,
    parent: usize,
    group_coeffs: &[Vec<Fp>],
    z: Fp,
    w: Fp,
) -> Fp {
    let kappa = source_params.need;
    let axes = target_params.segment_axes(parent);
    let selectors_z = axis_selectors(kappa, axes.rho_z, z);
    let selectors_w = axis_selectors(kappa, axes.rho_w, w);
    let mut acc = fp_zero();
    for u in 0..2 {
        for v in 0..2 {
            acc += selectors_z[u]
                * selectors_w[v]
                * eval_segment_poly(&group_coeffs[2 * u + v], kappa, z, w);
        }
    }
    acc
}

fn build_direct_stitched_claim(
    source_params: &CodedArrayParams,
    target_params: &CodedArrayParams,
    parent: usize,
    group_coeffs: &[Vec<Fp>],
    registry_index: usize,
) -> DirectClaim {
    let column_identifier = target_params.shard_registry[registry_index].z;
    let row_label = target_params.row_labels[registry_index];
    let stitched =
        |z, w| eval_stitched_poly(source_params, target_params, parent, group_coeffs, z, w);
    let column = w_axis_points(target_params, parent)
        .into_iter()
        .map(|eta| stitched(column_identifier, eta))
        .collect();
    let row = z_axis_points(target_params, parent)
        .into_iter()
        .map(|z| stitched(z, row_label))
        .collect();
    DirectClaim {
        column_identifier,
        row_label,
        column,
        row,
    }
}

/// Stitch the claim state of the four source segments of one group, given in
/// group order `2u + v`, entry by entry.
fn kappa_double_claim_via_stitch_formula(
    source_params: &CodedArrayParams,
    target_params: &CodedArrayParams,
    parent: usize,
    sources: &[DirectClaim],
) -> DirectClaim {
    assert_eq!(source_params.need * 2, target_params.need);
    assert_eq!(sources.len(), 4);
    let kappa = source_params.need;
    let column_identifier = sources[0].column_identifier;
    let row_label = sources[0].row_label;
    let axes = target_params.segment_axes(parent);
    let selectors_z = axis_selectors(kappa, axes.rho_z, column_identifier);
    let selectors_w = axis_selectors(kappa, axes.rho_w, row_label);
    let mut column = vec![fp_zero(); 2 * kappa];
    let mut row = vec![fp_zero(); 2 * kappa];
    for i in 0..kappa {
        for v in 0..2 {
            column[2 * i + v] =
                selectors_z[0] * sources[v].column[i] + selectors_z[1] * sources[2 + v].column[i];
        }
        for u in 0..2 {
            row[2 * i + u] =
                selectors_w[0] * sources[2 * u].row[i] + selectors_w[1] * sources[2 * u + 1].row[i];
        }
    }
    DirectClaim {
        column_identifier,
        row_label,
        column,
        row,
    }
}

/// Recover the claim state of the off-diagonal source segments `(0, 1)` and
/// `(1, 0)` from the stitched claim state and the claim state of the diagonal
/// source segments `(0, 0)` and `(1, 1)`.
fn kappa_halve_claims_via_formula(
    source_params: &CodedArrayParams,
    target_params: &CodedArrayParams,
    parent: usize,
    stitched: &DirectClaim,
    segment_00: &DirectClaim,
    segment_11: &DirectClaim,
) -> (DirectClaim, DirectClaim) {
    assert_eq!(source_params.need * 2, target_params.need);
    let kappa = source_params.need;
    let column_identifier = stitched.column_identifier;
    let row_label = stitched.row_label;
    let axes = target_params.segment_axes(parent);
    let selectors_z = axis_selectors(kappa, axes.rho_z, column_identifier);
    let selectors_w = axis_selectors(kappa, axes.rho_w, row_label);
    let inv_z = selectors_z.map(|selector| selector.inverse().expect("nonzero Z selector"));
    let inv_w = selectors_w.map(|selector| selector.inverse().expect("nonzero W selector"));
    let mut segment_01 = DirectClaim {
        column_identifier,
        row_label,
        column: Vec::with_capacity(kappa),
        row: Vec::with_capacity(kappa),
    };
    let mut segment_10 = segment_01.clone();
    for i in 0..kappa {
        segment_10
            .column
            .push((stitched.column[2 * i] - selectors_z[0] * segment_00.column[i]) * inv_z[1]);
        segment_01
            .column
            .push((stitched.column[2 * i + 1] - selectors_z[1] * segment_11.column[i]) * inv_z[0]);
        segment_01
            .row
            .push((stitched.row[2 * i] - selectors_w[0] * segment_00.row[i]) * inv_w[1]);
        segment_10
            .row
            .push((stitched.row[2 * i + 1] - selectors_w[1] * segment_11.row[i]) * inv_w[0]);
    }
    (segment_01, segment_10)
}

/// Store holding only the claim at `registry_index`, with every segment
/// encoded from the cells of its sampled polynomial.
fn build_single_claim_store(
    params: &CodedArrayParams,
    segments: usize,
    registry_index: usize,
) -> UnauthStore {
    let claim = ClaimBundle::exact_for_registry_index(params, registry_index).expect("claim");
    let mut store = UnauthStore::new_with_claims(params.clone(), vec![claim]);
    store
        .extend_with_segments((segments * params.segment_len) as u64, |seg| {
            direct_cells(params, seg, &sample_segment_coeffs(params.n_grid, seg))
                .into_iter()
                .map(encode_value)
                .collect()
        })
        .expect("extend");
    store
}

fn stored_claim(store: &UnauthStore, seg: usize, registry_index: usize) -> DirectClaim {
    let params = &store.params;
    let entry = params.shard_registry[registry_index];
    let row_label = params.row_labels[registry_index];
    DirectClaim {
        column_identifier: entry.z,
        row_label,
        column: store
            .coded
            .shard_segment(params, &entry.key, seg)
            .expect("stored column")
            .to_vec(),
        row: store
            .coded
            .row_segment(params, &RowKey::new(entry.key, row_label), seg)
            .expect("stored row")
            .to_vec(),
    }
}

fn recovered_state(claim: &DirectClaim, registry_index: usize) -> RecoveredClaimSegmentState {
    RecoveredClaimSegmentState {
        shards: vec![(registry_index, claim.column.clone())],
        rows: vec![(registry_index, claim.row.clone())],
    }
}

#[test]
fn stitch_formula_kappa_double_matches_direct_target_claim() {
    let params = build_params(4, 8);
    let kappa_doubled = build_kappa_double_params(&params).expect("kappa-double params");
    let registry_index = 3usize;
    let parents = 2usize;
    let mut store = build_single_claim_store(&params, 4 * parents, registry_index);

    let mut expected_by_parent = Vec::new();
    for parent in 0..parents {
        let group_coeffs: Vec<_> = (0..4)
            .map(|position| sample_segment_coeffs(params.n_grid, 4 * parent + position))
            .collect();

        // The stitched polynomial takes the same value as the source
        // polynomials at every public index of the group.
        let axes = kappa_doubled.segment_axes(parent);
        for logical_offset in 0..kappa_doubled.segment_len {
            let position = logical_offset / params.segment_len;
            let source_cells =
                direct_cells(&params, 4 * parent + position, &group_coeffs[position]);
            let source_grid_index =
                logical_offset_to_grid_index(&params, logical_offset % params.segment_len);
            let target_grid_index = logical_offset_to_grid_index(&kappa_doubled, logical_offset);
            let (z, w) = kappa_doubled.cell_points(&axes, target_grid_index);
            assert_eq!(
                eval_stitched_poly(&params, &kappa_doubled, parent, &group_coeffs, z, w),
                source_cells[source_grid_index]
            );
        }

        let sources: Vec<_> = (0..4)
            .map(|position| {
                let seg = 4 * parent + position;
                let source =
                    build_direct_claim(&params, seg, &group_coeffs[position], registry_index);
                assert_eq!(stored_claim(&store, seg, registry_index), source);
                source
            })
            .collect();
        let actual =
            kappa_double_claim_via_stitch_formula(&params, &kappa_doubled, parent, &sources);
        let expected = build_direct_stitched_claim(
            &params,
            &kappa_doubled,
            parent,
            &group_coeffs,
            registry_index,
        );
        assert_eq!(actual, expected);
        expected_by_parent.push(expected);
    }

    store.kappa_double().expect("kappa-double store");
    assert_eq!(store.segments(), parents);
    for (parent, expected) in expected_by_parent.iter().enumerate() {
        assert_eq!(&stored_claim(&store, parent, registry_index), expected);
    }
}

#[test]
fn kappa_halve_local_formula_matches_direct_source_claims() {
    let params = build_params(4, 8);
    let kappa_doubled = build_kappa_double_params(&params).expect("kappa-double params");
    let registry_index = 5usize;
    let parents = 2usize;
    let mut store = build_single_claim_store(&params, 4 * parents, registry_index);
    store.kappa_double().expect("kappa-double store");

    let mut source_halves = Vec::new();
    let mut expected_sources = Vec::new();
    for parent in 0..parents {
        let group_coeffs: Vec<_> = (0..4)
            .map(|position| sample_segment_coeffs(params.n_grid, 4 * parent + position))
            .collect();
        let sources: Vec<_> = (0..4)
            .map(|position| {
                build_direct_claim(
                    &params,
                    4 * parent + position,
                    &group_coeffs[position],
                    registry_index,
                )
            })
            .collect();
        let stitched = build_direct_stitched_claim(
            &params,
            &kappa_doubled,
            parent,
            &group_coeffs,
            registry_index,
        );
        let (segment_01, segment_10) = kappa_halve_claims_via_formula(
            &params,
            &kappa_doubled,
            parent,
            &stitched,
            &sources[0],
            &sources[3],
        );
        assert_eq!(segment_01, sources[1]);
        assert_eq!(segment_10, sources[2]);

        source_halves.push(SourceHalfState::Diagonal {
            segment_00: recovered_state(&sources[0], registry_index),
            segment_11: recovered_state(&sources[3], registry_index),
        });
        expected_sources.extend(sources);
    }

    store
        .kappa_halve(&source_halves)
        .expect("kappa-halve store");
    assert_eq!(store.segments(), expected_sources.len());
    for (seg, expected) in expected_sources.iter().enumerate() {
        assert_eq!(&stored_claim(&store, seg, registry_index), expected);
    }
}

#[test]
fn array_mapping_changes_with_epoch_and_kappa_halve_restores_aligned_blocks() {
    let params = build_params(4, 8);
    let kappa_doubled = build_kappa_double_params(&params).expect("kappa-double params");
    let kappa_halved = build_kappa_halve_params(&kappa_doubled).expect("kappa-halve params");
    let kappa = params.need;

    // Source segment `2u + v` of a group is the matching quarter of the
    // stitched segment in public order; its cell `(i, j)` moves to
    // `(2i + u, 2j + v)`.
    let mut changed = false;
    for logical_offset in 0..kappa_doubled.segment_len {
        let position = logical_offset / params.segment_len;
        let (u, v) = (position / 2, position % 2);
        let source_grid_index =
            logical_offset_to_grid_index(&params, logical_offset % params.segment_len);
        let (i, j) = (source_grid_index / kappa, source_grid_index % kappa);
        let stitched = logical_offset_to_grid_index(&kappa_doubled, logical_offset);
        assert_eq!(
            stitched,
            kappa_doubled.grid_index_for_axes(2 * i + u, 2 * j + v)
        );
        if stitched != logical_offset {
            changed = true;
        }
    }
    assert!(
        changed,
        "kappa-double should change the current array-to-grid map"
    );

    for logical_offset in 0..params.segment_len {
        let original = logical_offset_to_grid_index(&params, logical_offset);
        let restored = logical_offset_to_grid_index(&kappa_halved, logical_offset);
        assert_eq!(restored, original);
    }

    let len = 2 * params.segment_len as u64;
    for block in 0..(params.segment_len / params.s) {
        let current_block = block_grid_indices(&params, block);
        let mapped: Vec<_> = (0..params.s)
            .map(|offset| logical_offset_to_grid_index(&params, block * params.s + offset))
            .collect();
        assert_eq!(mapped, current_block);
        assert_eq!(block_grid_indices(&kappa_halved, block), current_block);
        assert_eq!(
            aligned_block_indices(&kappa_halved, block as u64, len),
            aligned_block_indices(&params, block as u64, len)
        );
    }
}
