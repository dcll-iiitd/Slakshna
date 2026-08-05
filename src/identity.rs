use bech32::{ self, Bech32, Hrp };
use ed25519_dalek::{ SigningKey, VerifyingKey, Signer, Signature };
use rand::rngs::OsRng;
use sha2::{ Sha256, Digest };
use serde::{ Deserialize, Serialize };
use std::fmt;

const NODE_ID_HRP: &str = "slakshna";

/// A node's stable federation identity, derived from its Ed25519 public key.
/// Every model update, peer review, and staged delta file is keyed by this id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn new(s: &str) -> Self {
        NodeId(s.to_string())
    }

    pub fn from_public_key(public_key: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(public_key);
        let hash = hasher.finalize();
        let hash_bytes = &hash[..20];

        let hrp = Hrp::parse(NODE_ID_HRP).unwrap();
        let encoded = bech32::encode::<Bech32>(hrp, hash_bytes).unwrap();
        NodeId(encoded)
    }

    pub fn is_valid(&self) -> bool {
        if !self.0.starts_with(NODE_ID_HRP) {
            return false;
        }
        bech32::decode(&self.0).is_ok()
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone)]
pub struct Keypair {
    pub signing_key: SigningKey,
    pub verifying_key: VerifyingKey,
}

impl Keypair {
    pub fn generate() -> Self {
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();

        Keypair {
            signing_key,
            verifying_key,
        }
    }

    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let signing_key = SigningKey::from_bytes(bytes);
        let verifying_key = signing_key.verifying_key();

        Ok(Keypair {
            signing_key,
            verifying_key,
        })
    }

    pub fn from_hex(hex_str: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let bytes = hex::decode(hex_str)?;
        if bytes.len() != 32 {
            return Err("Private key must be 32 bytes".into());
        }
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes);
        Self::from_bytes(&key_bytes)
    }

    pub fn node_id(&self) -> NodeId {
        NodeId::from_public_key(self.verifying_key.as_bytes())
    }

    pub fn public_key_hex(&self) -> String {
        hex::encode(self.verifying_key.as_bytes())
    }

    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        let signature = self.signing_key.sign(message);
        signature.to_bytes().to_vec()
    }

    pub fn sign_hex(&self, message: &[u8]) -> String {
        hex::encode(self.sign(message))
    }

    pub fn verify(&self, message: &[u8], signature: &[u8]) -> bool {
        if signature.len() != 64 {
            return false;
        }
        let sig_bytes: [u8; 64] = signature.try_into().unwrap();
        let sig = Signature::from_bytes(&sig_bytes);
        self.verifying_key.verify_strict(message, &sig).is_ok()
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }

    /// Seed for this node's Iroh transport key, derived from the persisted
    /// federation keypair so that the EndpointId peers dial stays the same
    /// across restarts. Without this the node would get a fresh EndpointId
    /// every process and every peer's cached address for it would go stale —
    /// which is exactly what a bootstrap node used to paper over.
    ///
    /// Domain-separated so the transport key is never the signing key itself.
    pub fn transport_seed(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(TRANSPORT_KEY_DOMAIN);
        hasher.update(self.signing_key.to_bytes());
        let hash = hasher.finalize();
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&hash);
        seed
    }
}

const TRANSPORT_KEY_DOMAIN: &[u8] = b"slakshna/iroh-endpoint/v1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let keypair = Keypair::generate();
        let node_id = keypair.node_id();
        assert!(node_id.is_valid());
        assert!(node_id.0.starts_with(NODE_ID_HRP));
    }

    #[test]
    fn test_transport_seed_is_stable_and_distinct() {
        let keypair = Keypair::generate();
        let reloaded = Keypair::from_bytes(&keypair.to_bytes()).unwrap();

        // Same persisted keypair must always yield the same Iroh EndpointId,
        // otherwise peers lose the node across restarts.
        assert_eq!(keypair.transport_seed(), reloaded.transport_seed());
        // ...but it must not be the signing key itself.
        assert_ne!(keypair.transport_seed(), keypair.to_bytes());
        assert_ne!(keypair.transport_seed(), Keypair::generate().transport_seed());
    }

    #[test]
    fn test_sign_and_verify() {
        let keypair = Keypair::generate();
        let message = b"Hello, SLAKSHNA!";
        let signature = keypair.sign(message);
        assert!(keypair.verify(message, &signature));
    }
}
