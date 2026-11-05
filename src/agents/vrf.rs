//! Verifiable random function over the node's Ed25519 key (A5).
//!
//! ECVRF construction after RFC 9381 §5 (edwards25519, SHA-512,
//! try-and-increment hash-to-curve, suite byte 0x03): the prover derives
//! Gamma = x·H(Y, alpha), proves knowledge of x with a Schnorr-style
//! Chaum–Pedersen proof (c, s), and the output is a hash of cofactor·Gamma.
//! Uniqueness comes from Gamma being fixed by (x, alpha), so a node cannot
//! grind alternative outputs for the same input.
//!
//! Interop note: integer/string conversions here use little-endian
//! throughout, and the implementation has not been cross-checked against the
//! RFC 9381 test vectors. It is self-consistent and sound as a VRF for
//! Slakshna's committees, but do not expect byte-compatibility with other
//! ECVRF libraries.

use curve25519_dalek::constants::ED25519_BASEPOINT_POINT as B;
use curve25519_dalek::edwards::{CompressedEdwardsY, EdwardsPoint};
use curve25519_dalek::scalar::{clamp_integer, Scalar};
use curve25519_dalek::traits::IsIdentity;
use sha2::{Digest, Sha512};

const SUITE: u8 = 0x03;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VrfProof {
    gamma: [u8; 32],
    c: [u8; 16],
    s: [u8; 32],
}

impl VrfProof {
    pub fn to_bytes(&self) -> [u8; 80] {
        let mut o = [0u8; 80];
        o[..32].copy_from_slice(&self.gamma);
        o[32..48].copy_from_slice(&self.c);
        o[48..].copy_from_slice(&self.s);
        o
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() != 80 {
            return None;
        }
        Some(VrfProof { gamma: b[..32].try_into().ok()?, c: b[32..48].try_into().ok()?, s: b[48..].try_into().ok()? })
    }

    pub fn to_hex(&self) -> String {
        hex::encode(self.to_bytes())
    }

    pub fn from_hex(s: &str) -> Option<Self> {
        Self::from_bytes(&hex::decode(s).ok()?)
    }
}

fn expand_secret(seed: &[u8; 32]) -> (Scalar, [u8; 32]) {
    let h = Sha512::digest(seed);
    let mut lo = [0u8; 32];
    lo.copy_from_slice(&h[..32]);
    let x = Scalar::from_bytes_mod_order(clamp_integer(lo));
    let mut prefix = [0u8; 32];
    prefix.copy_from_slice(&h[32..]);
    (x, prefix)
}

/// Public key (compressed point) for a 32-byte Ed25519 seed. Matches the
/// Ed25519 public key of the same seed.
pub fn public_key(seed: &[u8; 32]) -> [u8; 32] {
    let (x, _) = expand_secret(seed);
    (x * B).compress().to_bytes()
}

fn hash_to_curve(pk: &[u8; 32], alpha: &[u8]) -> EdwardsPoint {
    for ctr in 0u8..=255 {
        let mut h = Sha512::new();
        h.update([SUITE, 0x01]);
        h.update(pk);
        h.update(alpha);
        h.update([ctr, 0x00]);
        let d = h.finalize();
        let mut enc = [0u8; 32];
        enc.copy_from_slice(&d[..32]);
        if let Some(p) = CompressedEdwardsY(enc).decompress() {
            let p = p.mul_by_cofactor();
            if !p.is_identity() {
                return p;
            }
        }
    }
    // Probability ~2^-256; unreachable in practice.
    unreachable!("hash_to_curve exhausted 256 counters")
}

fn challenge(points: [&EdwardsPoint; 5]) -> [u8; 16] {
    let mut h = Sha512::new();
    h.update([SUITE, 0x02]);
    for p in points {
        h.update(p.compress().as_bytes());
    }
    h.update([0x00]);
    let d = h.finalize();
    let mut c = [0u8; 16];
    c.copy_from_slice(&d[..16]);
    c
}

fn c_scalar(c: &[u8; 16]) -> Scalar {
    let mut w = [0u8; 32];
    w[..16].copy_from_slice(c);
    Scalar::from_bytes_mod_order(w)
}

fn output(gamma: &EdwardsPoint) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update([SUITE, 0x03]);
    h.update(gamma.mul_by_cofactor().compress().as_bytes());
    h.update([0x00]);
    h.finalize().into()
}

/// Evaluate the VRF: returns the 64-byte output and the proof.
pub fn prove(seed: &[u8; 32], alpha: &[u8]) -> ([u8; 64], VrfProof) {
    let (x, prefix) = expand_secret(seed);
    let y = x * B;
    let pk = y.compress().to_bytes();
    let h = hash_to_curve(&pk, alpha);
    let gamma = x * h;
    // Deterministic nonce (RFC 8032 style): k = H(prefix || H_string)
    let mut nh = Sha512::new();
    nh.update(prefix);
    nh.update(h.compress().as_bytes());
    let k = Scalar::from_bytes_mod_order_wide(&nh.finalize().into());
    let c = challenge([&y, &h, &gamma, &(k * B), &(k * h)]);
    let s = k + c_scalar(&c) * x;
    (output(&gamma), VrfProof { gamma: gamma.compress().to_bytes(), c, s: s.to_bytes() })
}

/// Verify a proof for `(pk, alpha)`; returns the output on success.
pub fn verify(pk: &[u8; 32], alpha: &[u8], proof: &VrfProof) -> Option<[u8; 64]> {
    let y = CompressedEdwardsY(*pk).decompress()?;
    if y.mul_by_cofactor().is_identity() {
        return None; // small-order key
    }
    let gamma = CompressedEdwardsY(proof.gamma).decompress()?;
    let s = Scalar::from_canonical_bytes(proof.s).into_option()?;
    let h = hash_to_curve(pk, alpha);
    let c = c_scalar(&proof.c);
    let u = s * B - c * y;
    let v = s * h - c * gamma;
    if challenge([&y, &h, &gamma, &u, &v]) == proof.c {
        Some(output(&gamma))
    } else {
        None
    }
}

/// Map a VRF output to (0, 1]: used as the uniform draw in weighted sortition.
pub fn unit_interval(out: &[u8; 64]) -> f64 {
    let n = u64::from_le_bytes(out[..8].try_into().unwrap()) >> 11; // 53 bits
    (n as f64 + 1.0) / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::Keypair;

    #[test]
    fn public_key_matches_ed25519() {
        let kp = Keypair::generate();
        assert_eq!(public_key(&kp.to_bytes()), *kp.verifying_key.as_bytes());
    }

    #[test]
    fn prove_verify_roundtrip_and_determinism() {
        let kp = Keypair::generate();
        let seed = kp.to_bytes();
        let pk = public_key(&seed);
        let (out, proof) = prove(&seed, b"alpha");
        assert_eq!(verify(&pk, b"alpha", &proof), Some(out));
        let (out2, proof2) = prove(&seed, b"alpha");
        assert_eq!((out, proof.clone()), (out2, proof2), "VRF is deterministic");
        let (out3, _) = prove(&seed, b"beta");
        assert_ne!(out, out3);
    }

    #[test]
    fn rejects_wrong_input_key_and_tampered_proof() {
        let (a, b) = (Keypair::generate(), Keypair::generate());
        let (_, proof) = prove(&a.to_bytes(), b"x");
        assert!(verify(&public_key(&a.to_bytes()), b"y", &proof).is_none());
        assert!(verify(&public_key(&b.to_bytes()), b"x", &proof).is_none());
        for i in [0usize, 20, 40, 70] {
            let mut bytes = proof.to_bytes();
            bytes[i] ^= 1;
            if let Some(p) = VrfProof::from_bytes(&bytes) {
                assert!(verify(&public_key(&a.to_bytes()), b"x", &p).is_none(), "byte {i}");
            }
        }
        assert!(VrfProof::from_bytes(&[0u8; 79]).is_none());
    }

    #[test]
    fn small_order_key_and_noncanonical_s_rejected() {
        let kp = Keypair::generate();
        let (_, proof) = prove(&kp.to_bytes(), b"x");
        let mut identity = [0u8; 32];
        identity[0] = 1; // encoding of the neutral point (0, 1)
        assert!(verify(&identity, b"x", &proof).is_none());
        let mut bytes = proof.to_bytes();
        bytes[48..].copy_from_slice(&[0xff; 32]); // s >= group order
        let p = VrfProof::from_bytes(&bytes).unwrap();
        assert!(verify(&public_key(&kp.to_bytes()), b"x", &p).is_none());
    }

    #[test]
    fn outputs_look_uniform() {
        let kp = Keypair::generate();
        let n = 300;
        let mean = (0..n).map(|i| unit_interval(&prove(&kp.to_bytes(), format!("{i}").as_bytes()).0)).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.08, "mean {mean}");
    }

    #[test]
    fn hex_roundtrip() {
        let kp = Keypair::generate();
        let (_, p) = prove(&kp.to_bytes(), b"x");
        assert_eq!(VrfProof::from_hex(&p.to_hex()), Some(p));
    }
}
