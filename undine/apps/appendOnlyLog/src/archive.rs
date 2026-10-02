use crate::codec::CodecError;
use crate::mock::MockBatchError;
use crate::registry::{AnchorError, AppendOnlyLogAnchorFile};
use paper_a_optimized::net::auth::{AuthClient as NetAuthClient, ClientError as NetClientError};
use paper_a_optimized::unauth::build_params_for_kappa;
use std::path::{Path, PathBuf};

pub(crate) struct LoadedArchive {
    pub(crate) anchor_path: PathBuf,
    pub(crate) anchor: AppendOnlyLogAnchorFile,
    pub(crate) inner: NetAuthClient,
}

pub(crate) fn load_archive(path: &Path) -> Result<LoadedArchive, ArchiveError> {
    load_archive_opening_at(path, None)
}

/// Load the archive with a client that opens at threshold `open_kappa`, or
/// at the registry's threshold when `open_kappa` is `None`.
///
/// `open_kappa` is `2^j` times the registry's threshold and at most the
/// number of claims. The maintainers keep their stored state and stitch while
/// they serve.
pub(crate) fn load_archive_opening_at(
    path: &Path,
    open_kappa: Option<usize>,
) -> Result<LoadedArchive, ArchiveError> {
    let anchor = AppendOnlyLogAnchorFile::load(path).map_err(ArchiveError::Anchor)?;
    let registry = anchor.load_registry(path).map_err(ArchiveError::Anchor)?;
    let mut inner = NetAuthClient::from_registry(&registry).map_err(ArchiveError::Auth)?;
    if let Some(open_kappa) = open_kappa {
        let stored_kappa = inner.params.need;
        let claims = inner.params.shard_registry.len();
        let invalid = |reason: String| ArchiveError::InvalidOpenKappa {
            open_kappa,
            stored_kappa,
            reason,
        };
        if open_kappa > claims {
            return Err(invalid(format!(
                "it exceeds the {claims} claims of the registry"
            )));
        }
        inner.params = build_params_for_kappa(&inner.params, open_kappa)
            .map_err(|err| invalid(format!("{err:?}")))?;
    }
    if let Some(root) = anchor.trusted_root().map_err(ArchiveError::Anchor)? {
        inner.set_trusted_root(root);
    }
    Ok(LoadedArchive {
        anchor_path: path.to_path_buf(),
        anchor,
        inner,
    })
}

#[derive(Debug)]
pub enum ArchiveError {
    Anchor(AnchorError),
    Auth(NetClientError),
    Codec(CodecError),
    MockBatch(MockBatchError),
    Internal(String),
    MissingTrustedRoot,
    InvalidByteRange {
        offset: usize,
        len: usize,
    },
    HeightNotCommitted {
        height: u64,
        committed_leaves: u64,
    },
    LeafIndexNotCommitted {
        index: u64,
        committed_leaves: u64,
    },
    HeightMismatch {
        expected: u64,
        actual: u64,
    },
    MissingFrontierWitness,
    InvalidFrontierState(String),
    CommittedLengthMismatch {
        anchor_committed: u64,
        server_reported: u64,
    },
    CommittedLengthOverflow,
    InvalidOpenKappa {
        open_kappa: usize,
        stored_kappa: usize,
        reason: String,
    },
}

impl core::fmt::Display for ArchiveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Anchor(err) => write!(f, "{err}"),
            Self::Auth(err) => write!(f, "{err:?}"),
            Self::Codec(err) => write!(f, "{err}"),
            Self::MockBatch(err) => write!(f, "{err}"),
            Self::Internal(message) => write!(f, "{message}"),
            Self::MissingTrustedRoot => write!(
                f,
                "missing trusted root for a non-empty archive; pin one in the anchor file first"
            ),
            Self::InvalidByteRange { offset, len } => {
                write!(
                    f,
                    "invalid bulk_updates byte range offset={offset} len={len}"
                )
            }
            Self::HeightNotCommitted {
                height,
                committed_leaves,
            } => write!(
                f,
                "height {height} is not fully committed under committed_leaves={committed_leaves}"
            ),
            Self::LeafIndexNotCommitted {
                index,
                committed_leaves,
            } => write!(
                f,
                "leaf index {index} is not committed under committed_leaves={committed_leaves}"
            ),
            Self::HeightMismatch { expected, actual } => {
                write!(f, "expected next height {expected}, got {actual}")
            }
            Self::InvalidOpenKappa {
                open_kappa,
                stored_kappa,
                reason,
            } => write!(
                f,
                "cannot open at kappa {open_kappa} from the registry's kappa {stored_kappa}: {reason}"
            ),
            Self::MissingFrontierWitness => {
                write!(
                    f,
                    "missing cached frontier witness for a non-empty append frontier"
                )
            }
            Self::InvalidFrontierState(message) => write!(f, "{message}"),
            Self::CommittedLengthMismatch {
                anchor_committed,
                server_reported,
            } => write!(
                f,
                "anchor committed_leaves={anchor_committed} but server reported {server_reported}"
            ),
            Self::CommittedLengthOverflow => write!(f, "committed length overflow"),
        }
    }
}

impl std::error::Error for ArchiveError {}
