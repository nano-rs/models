//! Checks models.yaml and agent_defaults.yaml.
//!
//! The structs below mirror nano's catalog sync
//! (nanosiem-enterprise/src/melod/model_catalog_sync.rs: `ModelCatalogYaml`,
//! `CatalogModel`, `AgentDefaultsYaml`, `ProviderTiers`, `TierConfig`). If nano
//! changes those, change these in the same PR. On top of what nano itself
//! rejects, this also enforces the rules that keep the catalog safe to sync:
//! unique IDs, known providers and tiers, defaults that point at listed
//! models, no YAML anchors, and the plain line layout tools/sync relies on.
//!
//! Usage: catalog-check <repo root>

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::process::ExitCode;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ModelCatalogYaml {
    models: Vec<CatalogModel>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CatalogModel {
    model_id: String,
    display_name: String,
    #[serde(default)]
    context_window: Option<i32>,
    #[serde(default)]
    supports_temperature: Option<bool>,
    #[serde(default)]
    tier: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AgentDefaultsYaml {
    providers: HashMap<String, ProviderTiers>,
    agents: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ProviderTiers {
    #[serde(rename = "_default")]
    default: TierConfig,
    #[serde(default)]
    capable: Option<TierConfig>,
    #[serde(default)]
    fast: Option<TierConfig>,
    #[serde(default)]
    mid: Option<TierConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct TierConfig {
    model_id: String,
    max_tokens: i32,
    temperature: f32,
}

/// Provider prefixes nano's gateway can route (see `ai_gateway.rs`), mapped
/// to the `provider_credentials.provider` key used in agent_defaults.yaml.
const PREFIXES: &[(&str, &str)] = &[
    ("anthropic", "anthropic"),
    ("bedrock", "bedrock"),
    ("gemini", "google"),
    ("openai", "openai"),
    ("workers-ai", "workers-ai"),
];
const TIERS: &[&str] = &["economy", "standard", "full"];
const AGENT_TIERS: &[&str] = &["capable", "fast", "mid"];
const FIELDS: &[&str] = &["display_name", "context_window", "supports_temperature", "tier"];

fn check_layout(text: &str, errors: &mut Vec<String>) {
    let mut in_models = false;
    for (n, line) in text.lines().enumerate() {
        let n = n + 1;
        // nano parses the catalog under an expansion budget; keep it free of
        // anchors, aliases and merge keys so it never comes close.
        let code = !line.trim_start().starts_with('#');
        if code && (line.contains(": &") || line.contains("- &") || line.contains(": *") || line.contains("- *") || line.contains("<<:")) {
            errors.push(format!("models.yaml:{n}: YAML anchors, aliases and merge keys are not allowed"));
        }
        if !in_models {
            if line == "models:" {
                in_models = true;
            } else if !(line.is_empty() || line.starts_with('#')) {
                errors.push(format!("models.yaml:{n}: only comments may come before `models:`"));
            }
            continue;
        }
        let trimmed = line.trim_start();
        if line.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("  - model_id: ") {
            if rest.trim().is_empty() || rest.contains(' ') {
                errors.push(format!("models.yaml:{n}: model_id must be one plain token"));
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("    ") {
            let key = rest.split(':').next().unwrap_or("");
            if FIELDS.contains(&key) && rest[key.len()..].starts_with(": ") {
                continue;
            }
        }
        errors.push(format!(
            "models.yaml:{n}: unexpected line; use `  - model_id: <id>` then 4-space `key: value` fields ({})",
            FIELDS.join(", ")
        ));
    }
    if !in_models {
        errors.push("models.yaml: missing top-level `models:`".into());
    }
}

fn run(root: &str) -> Vec<String> {
    let mut errors = Vec::new();
    let read = |name: &str| std::fs::read_to_string(format!("{root}/{name}"));
    let models_text = match read("models.yaml") {
        Ok(t) => t,
        Err(e) => return vec![format!("models.yaml: {e}")],
    };
    check_layout(&models_text, &mut errors);

    // Exactly what nano does: parse, refuse an empty list, require provider/model.
    let catalog: ModelCatalogYaml = match serde_yaml::from_str(&models_text) {
        Ok(c) => c,
        Err(e) => {
            errors.push(format!("models.yaml: nano would reject it: {e}"));
            return errors;
        }
    };
    if catalog.models.is_empty() {
        errors.push("models.yaml: nano rejects an empty catalog".into());
    }
    let mut ids = HashSet::new();
    let mut providers_listed = HashSet::new();
    for m in &catalog.models {
        let Some((prefix, rest)) = m.model_id.split_once('/') else {
            errors.push(format!("{}: nano requires provider/model-name", m.model_id));
            continue;
        };
        match PREFIXES.iter().find(|(p, _)| *p == prefix) {
            Some((_, key)) => {
                providers_listed.insert(*key);
            }
            None => errors.push(format!("{}: unknown provider prefix `{prefix}`", m.model_id)),
        }
        if rest.is_empty() {
            errors.push(format!("{}: empty model name", m.model_id));
        }
        if !ids.insert(m.model_id.as_str()) {
            errors.push(format!("{}: listed twice", m.model_id));
        }
        if m.display_name.trim().is_empty() {
            errors.push(format!("{}: empty display_name", m.model_id));
        }
        if let Some(c) = m.context_window {
            if c <= 0 {
                errors.push(format!("{}: context_window must be positive", m.model_id));
            }
        }
        if let Some(t) = &m.tier {
            if !TIERS.contains(&t.as_str()) {
                errors.push(format!("{}: tier `{t}` is not one of {}", m.model_id, TIERS.join(", ")));
            }
        }
    }

    let defaults_text = match read("agent_defaults.yaml") {
        Ok(t) => t,
        Err(e) => {
            errors.push(format!("agent_defaults.yaml: {e}"));
            return errors;
        }
    };
    let defaults: AgentDefaultsYaml = match serde_yaml::from_str(&defaults_text) {
        Ok(d) => d,
        Err(e) => {
            errors.push(format!("agent_defaults.yaml: nano would reject it: {e}"));
            return errors;
        }
    };
    for (provider, tiers) in &defaults.providers {
        if !PREFIXES.iter().any(|(_, key)| key == provider) {
            errors.push(format!("agent_defaults.yaml: unknown provider `{provider}`"));
        }
        let all = [("_default", Some(&tiers.default)), ("capable", tiers.capable.as_ref()),
                   ("fast", tiers.fast.as_ref()), ("mid", tiers.mid.as_ref())];
        for (tier, cfg) in all {
            let Some(cfg) = cfg else { continue };
            if !ids.contains(cfg.model_id.as_str()) {
                errors.push(format!(
                    "agent_defaults.yaml: {provider}.{tier} uses `{}`, which is not in models.yaml (nano would treat it as deprecated)",
                    cfg.model_id
                ));
            }
            let owner = cfg.model_id.split('/').next().unwrap_or("");
            let owner_key = PREFIXES.iter().find(|(p, _)| *p == owner).map(|(_, k)| *k);
            if owner_key != Some(provider.as_str()) {
                errors.push(format!("agent_defaults.yaml: {provider}.{tier} uses a model from another provider: `{}`", cfg.model_id));
            }
            if cfg.max_tokens <= 0 {
                errors.push(format!("agent_defaults.yaml: {provider}.{tier} max_tokens must be positive"));
            }
        }
    }
    for (agent, tier) in &defaults.agents {
        if !AGENT_TIERS.contains(&tier.as_str()) {
            errors.push(format!("agent_defaults.yaml: agent `{agent}` has unknown tier `{tier}`"));
        }
    }
    for p in &providers_listed {
        if !defaults.providers.contains_key(*p) {
            errors.push(format!("agent_defaults.yaml: no defaults for provider `{p}`, which has models listed"));
        }
    }
    errors
}

fn main() -> ExitCode {
    let root = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let errors = run(&root);
    if errors.is_empty() {
        println!("catalog-check: models.yaml and agent_defaults.yaml are valid");
        ExitCode::SUCCESS
    } else {
        for e in &errors {
            eprintln!("error: {e}");
        }
        eprintln!("catalog-check: {} problem(s)", errors.len());
        ExitCode::FAILURE
    }
}
