//! Bench-local striped Reed-Solomon helpers.
//!
//! The fair external baseline for the unauth artifact is a conventional
//! storage stripe over the same field:
//! - one logical segment is split into codewords of dimension `k`
//! - one shard stores one evaluation from every codeword
//! - reads decode only the touched codeword(s)
//! - repair/onboarding reconstruct one shard vector across all codewords
//! - restriping decodes the full logical segment and re-encodes under a new `k`

use ark_ff::FftField;
use paper_a_optimized::unauth::{CodedArrayParams, barycentric_eval, precompute_barycentric};
use paper_a_optimized::{Fp, Value256, decode_value, encode_value};
use std::collections::HashMap;

/// Default symbol-layout label used in baseline metadata.
pub const SYMBOL_LAYOUT_ONE_PER_CODEWORD: &str = "one_symbol_per_codeword";

#[derive(Clone, Debug)]
pub struct StripedRsLayout {
    pub k: usize,
    pub n: usize,
    pub codeword_count: usize,
    pub logical_len: usize,
    pub source_points: Vec<Fp>,
    pub shard_points: Vec<Fp>,
}

#[derive(Clone, Debug)]
pub struct StripedRsSegment {
    pub logical_len: usize,
    pub codeword_count: usize,
    pub codewords: Vec<Vec<Fp>>,
    pub shard_vectors: Vec<Vec<Fp>>,
}

#[derive(Clone, Debug)]
pub struct StripedRsDecoder {
    helper_indices: Vec<usize>,
    helper_points: Vec<Fp>,
    helper_weights: paper_a_optimized::coded_array::rs::BarycentricWeights,
}

/// Build the fair striped-RS layout matching the artifact field and shard ids.
pub fn fair_layout_for_params(params: &CodedArrayParams, logical_len: usize) -> StripedRsLayout {
    let k = params.need;
    let n = params.shard_registry.len();
    assert!(k > 0, "baseline k must be positive");
    assert!(n >= k, "baseline n must cover k helper shards");
    let source_root = Fp::get_root_of_unity(k as u64).expect("baseline source root");
    let source_points = (0..k)
        .map(|idx| paper_a_optimized::fp_pow(source_root, idx as u64))
        .collect();
    let shard_points = params.shard_registry.iter().map(|entry| entry.a).collect();
    StripedRsLayout {
        k,
        n,
        codeword_count: logical_len.div_ceil(k),
        logical_len,
        source_points,
        shard_points,
    }
}

pub fn default_helper_indices(layout: &StripedRsLayout) -> Vec<usize> {
    (0..layout.k).collect()
}

pub fn encode_segment_from_values(
    layout: &StripedRsLayout,
    logical_values: &[Value256],
) -> StripedRsSegment {
    assert_eq!(
        logical_values.len(),
        layout.logical_len,
        "logical value length must match layout"
    );
    let mut codewords = Vec::with_capacity(layout.codeword_count);
    for chunk in logical_values.chunks(layout.k) {
        let mut codeword = vec![paper_a_optimized::fp_zero(); layout.k];
        for (slot, value) in codeword.iter_mut().zip(chunk.iter()) {
            *slot = decode_value(*value);
        }
        codewords.push(codeword);
    }
    let source_weights = precompute_barycentric(&layout.source_points);
    let mut shard_vectors =
        vec![vec![paper_a_optimized::fp_zero(); layout.codeword_count]; layout.n];
    for (codeword_idx, codeword) in codewords.iter().enumerate() {
        for (shard_idx, &shard_point) in layout.shard_points.iter().enumerate() {
            shard_vectors[shard_idx][codeword_idx] =
                barycentric_eval(&source_weights, codeword, shard_point);
        }
    }
    StripedRsSegment {
        logical_len: layout.logical_len,
        codeword_count: layout.codeword_count,
        codewords,
        shard_vectors,
    }
}

impl StripedRsLayout {
    pub fn decoder(&self, helper_indices: &[usize]) -> StripedRsDecoder {
        assert_eq!(
            helper_indices.len(),
            self.k,
            "fair striped-RS decode must use exactly k helpers"
        );
        for &idx in helper_indices {
            assert!(idx < self.n, "helper index out of range");
        }
        let helper_points = helper_indices
            .iter()
            .map(|&idx| self.shard_points[idx])
            .collect::<Vec<_>>();
        let helper_weights = precompute_barycentric(&helper_points);
        StripedRsDecoder {
            helper_indices: helper_indices.to_vec(),
            helper_points,
            helper_weights,
        }
    }
}

impl StripedRsDecoder {
    fn helper_codeword_samples(&self, segment: &StripedRsSegment, codeword_idx: usize) -> Vec<Fp> {
        assert!(
            codeword_idx < segment.codeword_count,
            "codeword index out of range"
        );
        self.helper_indices
            .iter()
            .map(|&idx| segment.shard_vectors[idx][codeword_idx])
            .collect()
    }

    pub fn decode_codeword(
        &self,
        layout: &StripedRsLayout,
        segment: &StripedRsSegment,
        codeword_idx: usize,
    ) -> Vec<Fp> {
        let helper_values = self.helper_codeword_samples(segment, codeword_idx);
        layout
            .source_points
            .iter()
            .map(|&source_point| {
                barycentric_eval(&self.helper_weights, &helper_values, source_point)
            })
            .collect()
    }

    pub fn read_point(
        &self,
        layout: &StripedRsLayout,
        segment: &StripedRsSegment,
        logical_offset: usize,
    ) -> Value256 {
        assert!(
            logical_offset < layout.logical_len,
            "logical offset out of range"
        );
        let codeword_idx = logical_offset / layout.k;
        let offset_in_codeword = logical_offset % layout.k;
        let helper_values = self.helper_codeword_samples(segment, codeword_idx);
        encode_value(barycentric_eval(
            &self.helper_weights,
            &helper_values,
            layout.source_points[offset_in_codeword],
        ))
    }

    pub fn read_block(
        &self,
        layout: &StripedRsLayout,
        segment: &StripedRsSegment,
        logical_offsets: &[usize],
    ) -> Vec<Value256> {
        let mut decoded = HashMap::<usize, Vec<Fp>>::new();
        logical_offsets
            .iter()
            .map(|&logical_offset| {
                assert!(
                    logical_offset < layout.logical_len,
                    "logical offset out of range"
                );
                let codeword_idx = logical_offset / layout.k;
                let offset_in_codeword = logical_offset % layout.k;
                let codeword = decoded
                    .entry(codeword_idx)
                    .or_insert_with(|| self.decode_codeword(layout, segment, codeword_idx));
                encode_value(codeword[offset_in_codeword])
            })
            .collect()
    }

    pub fn reconstruct_shard_vector(
        &self,
        layout: &StripedRsLayout,
        segment: &StripedRsSegment,
        target_index: usize,
    ) -> Vec<Fp> {
        assert!(target_index < layout.n, "target shard index out of range");
        self.reconstruct_shard_vector_at_point(segment, layout.shard_points[target_index])
    }

    pub fn reconstruct_shard_vector_at_point(
        &self,
        segment: &StripedRsSegment,
        target_point: Fp,
    ) -> Vec<Fp> {
        (0..segment.codeword_count)
            .map(|codeword_idx| {
                let helper_values = self.helper_codeword_samples(segment, codeword_idx);
                barycentric_eval(&self.helper_weights, &helper_values, target_point)
            })
            .collect()
    }

    pub fn decode_full_segment(
        &self,
        layout: &StripedRsLayout,
        segment: &StripedRsSegment,
    ) -> Vec<Value256> {
        let mut out = Vec::with_capacity(layout.logical_len);
        for codeword_idx in 0..segment.codeword_count {
            let codeword = self.decode_codeword(layout, segment, codeword_idx);
            let take = if codeword_idx + 1 == segment.codeword_count {
                layout.logical_len - codeword_idx * layout.k
            } else {
                layout.k
            };
            out.extend(codeword.into_iter().take(take).map(encode_value));
        }
        out
    }

    pub fn restripe_target_shards(
        &self,
        source_layout: &StripedRsLayout,
        source_segment: &StripedRsSegment,
        target_layout: &StripedRsLayout,
        target_indices: &[usize],
    ) -> Vec<Vec<Fp>> {
        let decoded = self.decode_full_segment(source_layout, source_segment);
        let target_segment = encode_segment_from_values(target_layout, &decoded);
        target_indices
            .iter()
            .map(|&target_index| {
                assert!(
                    target_index < target_layout.n,
                    "target shard index out of range"
                );
                target_segment.shard_vectors[target_index].clone()
            })
            .collect()
    }

    pub fn helper_points(&self) -> &[Fp] {
        &self.helper_points
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SYMBOL_LAYOUT_ONE_PER_CODEWORD, default_helper_indices, encode_segment_from_values,
        fair_layout_for_params,
    };
    use paper_a_optimized::unauth::{
        CodedArrayConfig, build_coded_params, build_shard_registry_deterministic,
    };
    use paper_a_optimized::{Value256, encode_value};

    fn sample_params(segment_len: usize, s: usize) -> paper_a_optimized::CodedArrayParams {
        let n_grid = segment_len;
        let shard_count = (n_grid - 1) / s + 1;
        build_coded_params(&CodedArrayConfig {
            segment_len,
            s,
            shard_count,
            shard_registry: build_shard_registry_deterministic(shard_count, s, n_grid),
            party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
            seed: 7,
        })
    }

    #[test]
    fn round_trips_point_block_and_repair() {
        let params = sample_params(128, 16);
        let values: Vec<Value256> = (0..params.segment_len)
            .map(|idx| encode_value(paper_a_optimized::Fp::from((idx + 1) as u64)))
            .collect();
        let layout = fair_layout_for_params(&params, values.len());
        assert_eq!(SYMBOL_LAYOUT_ONE_PER_CODEWORD, "one_symbol_per_codeword");
        let segment = encode_segment_from_values(&layout, &values);
        let decoder = layout.decoder(&default_helper_indices(&layout));

        assert_eq!(decoder.read_point(&layout, &segment, 11), values[11]);
        assert_eq!(
            decoder.read_block(&layout, &segment, &[0, 1, 2, 15, 16]),
            vec![values[0], values[1], values[2], values[15], values[16]]
        );
        assert_eq!(
            decoder.reconstruct_shard_vector(&layout, &segment, 0),
            segment.shard_vectors[0]
        );

        let target_layout = fair_layout_for_params(&sample_params(128, 8), values.len());
        let restriped = decoder.restripe_target_shards(&layout, &segment, &target_layout, &[0]);
        let expected_target = encode_segment_from_values(&target_layout, &values);
        assert_eq!(restriped[0], expected_target.shard_vectors[0]);
    }
}
