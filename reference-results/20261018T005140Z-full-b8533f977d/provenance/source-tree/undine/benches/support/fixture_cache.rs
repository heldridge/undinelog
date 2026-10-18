//! Immutable, content-checked inputs for untimed benchmark setup.
//!
//! The executable hash isolates code changes. Callers supply every workload
//! parameter affecting the fixture, excluding trial counts and timed metrics.
//! No measured operation or mutable trial output belongs in this cache.

#![allow(dead_code)]

use serde::{Serialize, de::DeserializeOwned};
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub struct FixtureCache(Option<PathBuf>);

fn digest(path: &Path) -> (u64, String) {
    let mut file = fs::File::open(path).expect("open fixture for hashing");
    let mut hash = blake3::Hasher::new();
    let mut bytes = 0;
    let mut buffer = vec![0; 1 << 20];
    loop {
        let count = file.read(&mut buffer).expect("hash fixture");
        if count == 0 {
            break;
        }
        bytes += count as u64;
        hash.update(&buffer[..count]);
    }
    (bytes, hash.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    #[test]
    fn incomplete_and_corrupt_fixtures_cannot_be_consumed() {
        use super::*;
        let path = std::env::temp_dir().join(format!(
            "undine-fixture-test-{}-{}",
            std::process::id(),
            module_path!().replace("::", "-")
        ));
        if path.exists() {
            fs::remove_dir_all(&path).unwrap();
        }
        fs::create_dir_all(&path).unwrap();
        let cache = FixtureCache(Some(path.clone()));
        fs::write(path.join("data.bin"), b"incomplete write").unwrap();
        assert!(cache.load::<Vec<u64>>().is_none());
        cache.store(&vec![1u64, 5, 19]);
        assert_eq!(cache.load::<Vec<u64>>().unwrap(), vec![1, 5, 19]);
        fs::write(path.join("data.bin"), b"corrupt data").unwrap();
        assert!(std::panic::catch_unwind(|| cache.load::<Vec<u64>>()).is_err());
        fs::remove_dir_all(path).unwrap();
    }
}

impl FixtureCache {
    pub fn enabled(&self) -> bool {
        self.0.is_some()
    }

    pub fn new(kind: &str, workload: &impl Serialize) -> Self {
        let Some(root) = std::env::var_os("BENCH_FIXTURE_DIR") else {
            return Self(None);
        };
        let executable = std::env::current_exe().expect("benchmark executable");
        let mut hash = blake3::Hasher::new();
        hash.update(b"benchmark-fixture-v1\0");
        hash.update(digest(&executable).1.as_bytes());
        hash.update(kind.as_bytes());
        hash.update(&serde_json::to_vec(workload).expect("fixture parameters"));
        let path = PathBuf::from(root).join(hash.finalize().to_hex().as_str());
        fs::create_dir_all(&path).expect("fixture directory");
        Self(Some(path))
    }

    pub fn load<T: DeserializeOwned>(&self) -> Option<T> {
        self.load_with(|input| {
            bincode::serde::decode_from_std_read(input, bincode::config::standard())
                .expect("decode benchmark fixture")
        })
    }

    pub fn load_with<T>(&self, read: impl FnOnce(&mut BufReader<fs::File>) -> T) -> Option<T> {
        let path = self.0.as_ref()?;
        let metadata = path.join("complete.json");
        if !metadata.exists() {
            return None;
        }
        let expected: (u64, String) =
            serde_json::from_slice(&fs::read(metadata).expect("fixture metadata"))
                .expect("fixture checksum");
        let data = path.join("data.bin");
        assert_eq!(
            digest(&data),
            expected,
            "benchmark fixture changed or is truncated"
        );
        let mut input =
            BufReader::with_capacity(1 << 20, fs::File::open(data).expect("open fixture"));
        let result = read(&mut input);
        assert_eq!(
            input.read(&mut [0u8; 1]).expect("fixture EOF"),
            0,
            "trailing fixture bytes"
        );
        eprintln!("fixture cache hit: {}", path.display());
        Some(result)
    }

    pub fn store(&self, value: &impl Serialize) {
        self.store_with(|output| {
            bincode::serde::encode_into_std_write(value, output, bincode::config::standard())
                .expect("encode benchmark fixture");
        });
    }

    pub fn store_with(&self, write: impl FnOnce(&mut BufWriter<fs::File>)) {
        let Some(path) = &self.0 else {
            return;
        };
        let pending = path.join(format!("data-{}.tmp", std::process::id()));
        let mut output =
            BufWriter::with_capacity(1 << 20, fs::File::create(&pending).expect("create fixture"));
        write(&mut output);
        output.flush().expect("flush fixture");
        output.get_ref().sync_all().expect("sync fixture");
        drop(output);
        let checksum = digest(&pending);
        fs::rename(&pending, path.join("data.bin")).expect("install fixture");
        let metadata = path.join(format!("complete-{}.tmp", std::process::id()));
        fs::write(&metadata, serde_json::to_vec(&checksum).unwrap())
            .expect("write fixture checksum");
        fs::rename(metadata, path.join("complete.json")).expect("complete fixture");
        eprintln!("fixture cache write: {}", path.display());
    }
}
