use bech32::{ self, Bech32, Hrp };
use ed25519_dalek::{ SigningKey, VerifyingKey, Signer, Signature };
use rand::rngs::OsRng;
use sha2::{ Sha256, Digest };
use serde::{ Deserialize, Serialize };
use std::fmt;

const ADDRESS_HRP: &str = "iiitd1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Address(pub String);

impl Address {
    pub fn new(s: &str) -> Self {
        Address(s.to_string())
    }

    pub fn from_public_key(public_key: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(public_key);
        let hash = hasher.finalize();
        let hash_bytes = &hash[..20];

        let hrp = Hrp::parse(ADDRESS_HRP).unwrap();
        let encoded = bech32::encode::<Bech32>(hrp, hash_bytes).unwrap();
        Address(encoded)
    }

    pub fn is_valid(&self) -> bool {
        if !self.0.starts_with(ADDRESS_HRP) {
            return false;
        }
        // Contract and token addresses use hex format, not bech32
        if self.0.starts_with("iiitd1contract") || self.0.starts_with("iiitd1token") {
            return self.0.len() > 12 &&
                self.0
                    .chars()
                    .skip(4)
                    .all(|c| c.is_ascii_alphanumeric());
        }
        // Special addresses
        if self.0 == "iiitd1faucet" {
            return true;
        }
        bech32::decode(&self.0).is_ok()
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Address {
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

    pub fn address(&self) -> Address {
        Address::from_public_key(self.verifying_key.as_bytes())
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let keypair = Keypair::generate();
        let address = keypair.address();
        assert!(address.is_valid());
        assert!(address.0.starts_with(ADDRESS_HRP));
    }

    #[test]
    fn test_sign_and_verify() {
        let keypair = Keypair::generate();
        let message = b"Hello, IIITD!";
        let signature = keypair.sign(message);
        assert!(keypair.verify(message, &signature));
    }
}
