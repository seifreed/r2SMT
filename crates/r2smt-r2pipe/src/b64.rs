//! Tiny standards-compliant Base64 encoder.
//!
//! Lives here so the r2 adapter can pass arbitrary annotation text
//! through `CCu base64:<payload>` without adding a third-party
//! base64 crate to the workspace dependency tree.

const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode `input` as standard Base64 with `=` padding.
#[must_use]
pub fn encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    let mut chunks = input.chunks_exact(3);
    for chunk in chunks.by_ref() {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]);
        out.push(char::from(TABLE[((n >> 18) & 0x3F) as usize]));
        out.push(char::from(TABLE[((n >> 12) & 0x3F) as usize]));
        out.push(char::from(TABLE[((n >> 6) & 0x3F) as usize]));
        out.push(char::from(TABLE[(n & 0x3F) as usize]));
    }
    let rem = chunks.remainder();
    match rem.len() {
        1 => {
            let n = u32::from(rem[0]) << 16;
            out.push(char::from(TABLE[((n >> 18) & 0x3F) as usize]));
            out.push(char::from(TABLE[((n >> 12) & 0x3F) as usize]));
            out.push('=');
            out.push('=');
        }
        2 => {
            let n = (u32::from(rem[0]) << 16) | (u32::from(rem[1]) << 8);
            out.push(char::from(TABLE[((n >> 18) & 0x3F) as usize]));
            out.push(char::from(TABLE[((n >> 12) & 0x3F) as usize]));
            out.push(char::from(TABLE[((n >> 6) & 0x3F) as usize]));
            out.push('=');
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_empty_output() {
        assert_eq!(encode(b""), "");
    }

    #[test]
    fn rfc4648_vectors_round_trip() {
        assert_eq!(encode(b"f"), "Zg==");
        assert_eq!(encode(b"fo"), "Zm8=");
        assert_eq!(encode(b"foo"), "Zm9v");
        assert_eq!(encode(b"foob"), "Zm9vYg==");
        assert_eq!(encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn full_byte_range_round_trips() {
        let bytes: Vec<u8> = (0u8..=255).collect();
        let encoded = encode(&bytes);
        assert!(encoded.bytes().all(|b| TABLE.contains(&b) || b == b'='));
        let padding: usize = encoded.bytes().filter(|b| *b == b'=').count();
        assert!(padding <= 2);
        assert_eq!(encoded.len(), bytes.len().div_ceil(3) * 4);
    }
}
