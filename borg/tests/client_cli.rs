use std::process::Command;

#[test]
fn client_cli_accepts_max_errors_flag_before_loading_registry() {
    let output = Command::new(env!("CARGO_BIN_EXE_client"))
        .args([
            "--mode",
            "unauth",
            "--op",
            "repair",
            "--registry",
            "/definitely/missing/registry.json",
            "--segment",
            "0",
            "--registry-index",
            "0",
            "--max-errors",
            "0",
        ])
        .output()
        .expect("run client binary");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to load registry"),
        "stderr did not show registry loading after parsing --max-errors:\n{stderr}"
    );
}

#[test]
fn client_cli_rejects_invalid_max_errors() {
    let output = Command::new(env!("CARGO_BIN_EXE_client"))
        .args(["--mode", "unauth", "--op", "repair", "--max-errors", "nope"])
        .output()
        .expect("run client binary");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("invalid --max-errors"),
        "stderr did not reject invalid --max-errors:\n{stderr}"
    );
}
