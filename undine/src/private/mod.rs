//! Private openings of one position.
//!
//! A maintainer answers a private opening from its primary columns alone. This
//! module defines the database a maintainer serves and the request shape a
//! client uses, both derived from public parameters:
//! - [`layout`]: records, patterns, and where each cell of an opening sits;
//! - [`records`]: the stored entries of a record;
//! - [`pir`]: the PIR database over the records, its preprocessing, the
//!   answer to a query, and the client's query and decoding;
//! - [`opening`]: the request and reply of one private opening, the
//!   maintainer's handler, and the client's decoding into public shares;
//! - [`wire`]: the wire form of the request and the reply;
//! - [`bulk`]: the primary columns of a large log, built in a stream.

pub mod bulk;
pub mod keeper;
pub mod layout;
pub mod opening;
pub mod pir;
pub mod records;
pub mod wire;

pub use bulk::{
    BulkColumns, BulkLog, build_primary_columns, build_primary_columns_in_chunks,
    build_tree_cells, bulk_leaf_value,
};
pub use layout::{Grouping, LayoutError, PlannedEntry, PrivateLayout, pattern_bits};
pub use keeper::PrivateKeeper;
pub use opening::{
    PlannedUpdate, PrivateOpening, PrivateReadReply, PrivateReadRequest, PrivateReadResponse,
    PrivateState, release_at,
};
pub use pir::{
    PackKeys, PirClient, PirDatabase, PirError, PirShape, PreprocessTimes, TableDatabase,
    UpdateTimes, entries_to_cells, entry_from_cells,
};
pub use records::{local_claims, record_entries};
