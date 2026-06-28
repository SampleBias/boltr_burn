//! Boltr_Burn development CLI.

use std::path::PathBuf;

use anyhow::{Context, Result};
use boltr_backend_burn::{probe_backends, BackendProbe};
use boltr_backend_core::Boltz2Hparams;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "boltr-burn", about = "Burn backend dev tools for Boltz-2 inference")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Report compiled Burn backends and environment
    Doctor {
        #[arg(long)]
        json: bool,
    },
    /// Strict safetensors key check against Burn skeleton (or tch with --full-tch)
    VerifyWeights {
        path: PathBuf,
        #[arg(long, default_value_t = 384)]
        token_s: i64,
        #[arg(long, default_value_t = 128)]
        token_z: i64,
        #[arg(long)]
        blocks: Option<i64>,
        #[arg(long)]
        bond_type_feature: bool,
        #[arg(long)]
        partition: bool,
        #[arg(long)]
        reject_unused_file_keys: bool,
    },
    /// Load and print resolved hparams from JSON
    Hparams {
        path: PathBuf,
    },
    /// Run end-to-end predict on a fixture YAML (Phase 2+)
    Predict {
        fixture: PathBuf,
        #[arg(long, default_value = "cpu")]
        device: String,
    },
    /// A/B tch vs burn on the same collate fixture (Phase 1+)
    CompareTch {
        fixture: PathBuf,
    },
    /// Run opt-in module golden vs Python (Phase 1+)
    Golden {
        module: String,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Doctor { json } => cmd_doctor(json),
        Commands::VerifyWeights {
            path,
            token_s,
            token_z,
            blocks,
            bond_type_feature,
            partition,
            reject_unused_file_keys,
        } => cmd_verify_weights(
            path,
            token_s,
            token_z,
            blocks,
            bond_type_feature,
            partition,
            reject_unused_file_keys,
        ),
        Commands::Hparams { path } => cmd_hparams(path),
        Commands::Predict { fixture, device } => {
            anyhow::bail!(
                "predict not yet implemented for {}; device={device} (Phase 2)",
                fixture.display()
            )
        }
        Commands::CompareTch { fixture } => {
            anyhow::bail!(
                "compare-tch not yet implemented for {} (Phase 1)",
                fixture.display()
            )
        }
        Commands::Golden { module } => {
            anyhow::bail!("golden not yet implemented for module {module} (Phase 1)")
        }
    }
}

fn cmd_doctor(json: bool) -> Result<()> {
    let probe = probe_backends();
    let cache = dirs::home_dir()
        .map(|h| h.join(".cache/boltr"))
        .unwrap_or_default();
    let report = DoctorReport {
        version: env!("CARGO_PKG_VERSION"),
        rust_edition: "2021",
        backends: probe,
        boltr_cache_dir: cache.display().to_string(),
        weights: WeightStatus {
            conf: cache.join("boltz2_conf.safetensors").exists(),
            aff: cache.join("boltz2_aff.safetensors").exists(),
            hparams: cache.join("boltz2_hparams.json").exists(),
        },
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("boltr-burn doctor");
        println!("  version: {}", report.version);
        println!("  default backend feature: {}", report.backends.default_feature);
        println!("  ndarray compiled: {}", report.backends.ndarray_available);
        println!("  cuda compiled: {}", report.backends.cuda_compiled);
        println!("  wgpu compiled: {}", report.backends.wgpu_compiled);
        println!("  cache dir: {}", report.boltr_cache_dir);
        println!(
            "  boltz2_conf.safetensors: {}",
            if report.weights.conf { "found" } else { "missing" }
        );
        println!(
            "  boltz2_aff.safetensors: {}",
            if report.weights.aff { "found" } else { "missing" }
        );
        println!(
            "  boltz2_hparams.json: {}",
            if report.weights.hparams { "found" } else { "missing" }
        );
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct DoctorReport {
    version: &'static str,
    rust_edition: &'static str,
    backends: BackendProbe,
    boltr_cache_dir: String,
    weights: WeightStatus,
}

#[derive(serde::Serialize)]
struct WeightStatus {
    conf: bool,
    aff: bool,
    hparams: bool,
}

fn cmd_hparams(path: PathBuf) -> Result<()> {
    let raw = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let h = Boltz2Hparams::from_json_slice(&raw)?;
    println!("token_s: {}", h.resolved_token_s());
    println!("token_z: {}", h.resolved_token_z());
    println!(
        "pairformer blocks: {:?}",
        h.resolved_num_pairformer_blocks()
    );
    println!("bond_type_feature: {}", h.resolved_bond_type_feature());
    Ok(())
}

fn cmd_verify_weights(
    path: PathBuf,
    token_s: i64,
    token_z: i64,
    blocks: Option<i64>,
    bond_type_feature: bool,
    partition: bool,
    reject_unused_file_keys: bool,
) -> Result<()> {
    let bin = std::env::current_exe()?;
    let verify = bin
        .parent()
        .context("exe parent")?
        .join("verify_boltz2_burn_weights");

    let mut cmd = std::process::Command::new(verify);
    cmd.arg("--token-s").arg(token_s.to_string());
    cmd.arg("--token-z").arg(token_z.to_string());
    if let Some(b) = blocks {
        cmd.arg("--blocks").arg(b.to_string());
    }
    if bond_type_feature {
        cmd.arg("--bond-type-feature");
    }
    if partition {
        cmd.arg("--partition");
    }
    if reject_unused_file_keys {
        cmd.arg("--reject-unused-file-keys");
    }
    cmd.arg(path);

    let status = cmd.status().context("run verify_boltz2_burn_weights")?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}
