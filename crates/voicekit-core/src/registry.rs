use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub repo: String,
    #[serde(default)]
    pub url: String,
    pub path: String,
    pub files: Vec<String>,
    #[serde(default)]
    pub sha256: HashMap<String, String>,
}

impl ModelEntry {
    /// Absolute path to this model's directory given the models root (e.g. /models).
    pub fn abs_path(&self, models_root: &Path) -> PathBuf {
        models_root.join(&self.path)
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelRegistry {
    pub models: HashMap<String, ModelEntry>,
}

impl ModelRegistry {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading registry {:?}", path.as_ref()))?;
        toml::from_str(&raw).context("parsing models.toml")
    }

    /// Resolve the absolute model path for a named registry entry.
    pub fn resolve(&self, key: &str, models_root: &Path) -> Result<PathBuf> {
        self.models
            .get(key)
            .map(|e| e.abs_path(models_root))
            .ok_or_else(|| anyhow::anyhow!("model '{}' not found in registry", key))
    }
}