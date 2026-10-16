//! Independent repetition counts supplied by the artifact scheduler.

use std::collections::BTreeMap;

#[derive(Clone)]
pub struct Sampling {
    fallback: usize,
    pub groups: Option<BTreeMap<String, usize>>,
}

impl Sampling {
    pub fn new(fallback: usize) -> Self {
        let groups = std::env::var("BENCH_SAMPLE_GROUPS").ok().map(|raw| {
            let groups: BTreeMap<String, usize> =
                serde_json::from_str(&raw).expect("sample group counts");
            assert_eq!(
                groups.values().copied().max(),
                Some(fallback),
                "maximum invocation count"
            );
            groups
        });
        Self { fallback, groups }
    }

    pub fn count(&self, group: &str) -> usize {
        match &self.groups {
            None => self.fallback,
            Some(groups) => groups
                .get(group)
                .or_else(|| groups.get("all"))
                .copied()
                .unwrap_or(0),
        }
    }
}
