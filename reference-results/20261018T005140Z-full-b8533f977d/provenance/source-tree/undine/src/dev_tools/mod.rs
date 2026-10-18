pub mod auth_system;

pub use auth_system::{AuthTestSystem, sanity_dump, setup, setup_with_timing};

pub type System = AuthTestSystem;
