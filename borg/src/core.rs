//! Core domain types shared across layers.

use serde::{Deserialize, Serialize};

/// Linearization order for mapping Merkle nodes into the coded array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeLinearization {
    Preorder,
}

impl Default for NodeLinearization {
    fn default() -> Self {
        NodeLinearization::Preorder
    }
}

#[cfg(test)]
mod tests {
    use super::NodeLinearization;

    #[test]
    fn node_linearization_default_is_preorder() {
        assert_eq!(NodeLinearization::default(), NodeLinearization::Preorder);
    }

    #[test]
    fn node_linearization_serde_round_trip() {
        let val = NodeLinearization::Preorder;
        let encoded = serde_json::to_string(&val).expect("serialize NodeLinearization");
        assert_eq!(encoded, "\"preorder\"");
        let decoded: NodeLinearization =
            serde_json::from_str(&encoded).expect("deserialize NodeLinearization");
        assert_eq!(decoded, val);
    }
}
