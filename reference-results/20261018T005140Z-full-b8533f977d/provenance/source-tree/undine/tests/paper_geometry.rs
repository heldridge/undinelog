use paper_a_optimized::{PaperGeometryError, derive_paper_geometry};

#[test]
fn paper_geometry_derives_square_grid_from_kappa() {
    let geometry = derive_paper_geometry(8).expect("paper geometry");

    assert_eq!(geometry.segment_len, 64);
    assert_eq!(geometry.n_grid, 64);
    assert_eq!(geometry.shard_len, 8);
    assert_eq!(geometry.s(), 8);
    assert_eq!(geometry.kappa, 8);
    assert_eq!(geometry.need(), 8);
    assert_eq!(geometry.rows_per_claim, 1);
}

#[test]
fn paper_geometry_rejects_non_power_of_two_kappa() {
    let err = derive_paper_geometry(6).expect_err("not a power of two");
    assert_eq!(err, PaperGeometryError::KappaNotPowerOfTwo);
}

#[test]
fn paper_geometry_rejects_zero_kappa() {
    let err = derive_paper_geometry(0).expect_err("kappa must be positive");
    assert_eq!(err, PaperGeometryError::ZeroKappa);
}

#[test]
fn paper_geometry_rejects_kappa_whose_grid_overflows() {
    let kappa = 1usize << (usize::BITS / 2);
    let err = derive_paper_geometry(kappa).expect_err("kappa^2 must fit in usize");
    assert_eq!(err, PaperGeometryError::GridOverflow);
}
