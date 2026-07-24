//! Versioned content hashing shared by Volt's derived representations.
//!
//! The namespace acts as a domain and schema separator. A zero byte prevents a
//! namespace suffix from becoming ambiguous with the first serialized byte.

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{CogError, CogResult};

pub(crate) fn canonical_hash<T: Serialize>(namespace: &str, value: &T) -> CogResult<String> {
    let encoded =
        serde_json::to_vec(value).map_err(|error| CogError::Serialization(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(namespace.as_bytes());
    hasher.update([0]);
    hasher.update(encoded);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}")
            .map_err(|error| CogError::Serialization(error.to_string()))?;
    }
    Ok(output)
}
