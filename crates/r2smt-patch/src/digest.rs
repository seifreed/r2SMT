//! SHA-256 wrapper used by [`crate::manifest`] to record before / after
//! integrity hashes of the binary being patched.

use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::path::Path;

use sha2::{Digest, Sha256};

const READ_CHUNK: usize = 64 * 1024;

/// Compute the SHA-256 of a file, returning its lower-case hex digest.
///
/// Streams the file in 64 KiB chunks so the host memory footprint is
/// bounded regardless of binary size.
///
/// # Errors
///
/// Propagates [`std::io::Error`] if the file cannot be opened or read.
pub fn sha256_hex(path: impl AsRef<Path>) -> IoResult<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; READ_CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn sha256_empty_file_matches_known_constant() {
        let tmp = NamedTempFile::new().unwrap();
        let digest = sha256_hex(tmp.path()).unwrap();
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_known_vector_for_abc() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"abc").unwrap();
        tmp.flush().unwrap();
        let digest = sha256_hex(tmp.path()).unwrap();
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
