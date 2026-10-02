use paper_a_optimized::{PaperGeometryError, derive_paper_geometry};

#[test]
fn paper_geometry_derives_power_of_two_grid_artifact_params() {
    let geometry = derive_paper_geometry(128, 8).expect("paper geometry");

    assert_eq!(geometry.segment_len, 128);
    assert_eq!(geometry.n_grid, 128);
    assert_eq!(geometry.shard_len, 16);
    assert_eq!(geometry.s(), 16);
    assert_eq!(geometry.kappa, 8);
    assert_eq!(geometry.need(), 8);
    assert_eq!(geometry.rows_per_claim, 2);
}

#[test]
fn paper_geometry_rejects_non_power_of_two_grid_sizes() {
    let err = derive_paper_geometry(127, 8).expect_err("not a power of two");
    assert_eq!(err, PaperGeometryError::NonPowerOfTwoGridSize);
}

#[test]
fn paper_geometry_rejects_bad_kappa_divisibility() {
    let err = derive_paper_geometry(128, 7).expect_err("kappa must divide grid");
    assert_eq!(err, PaperGeometryError::KappaDoesNotDivideGrid);
}
