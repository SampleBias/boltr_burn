//! Compare a `.safetensors` file against the Burn Boltz2 skeleton parameter registry.

use std::path::PathBuf;

use boltr_backend_core::{
    list_safetensor_names, partition_safetensors_keys_for_inference,
    BOLTZ2_INFERENCE_TOP_LEVEL_KEYS,
};
use boltr_backend_burn::{Boltz2BurnModel, Boltz2BurnModelConfig};
use burn::backend::NdArray;

fn usage() -> ! {
    eprintln!(
        "\
Usage: verify_boltz2_burn_weights [OPTIONS] <PATH.safetensors>

Options:
  --token-s N           sequence width (default 384)
  --token-z N           pair width (default 128)
  --blocks N            pairformer depth hint (default 4)
  --bond-type-feature   match checkpoints with bond_type_feature=true
  --partition           print inference vs other key counts
  --reject-unused-file-keys   exit 1 if file has tensors not in skeleton registry

For full VarStore parity use Boltr's verify_boltz2_safetensors (requires LibTorch):
  cargo run -p boltr-backend-tch --features tch-backend --bin verify_boltz2_safetensors -- PATH
"
    );
    std::process::exit(2);
}

fn main() {
    let mut token_s = 384_i64;
    let mut token_z = 128_i64;
    let mut blocks: Option<i64> = Some(4);
    let mut bond_type = false;
    let mut partition = false;
    let mut reject_unused = false;
    let mut path: Option<PathBuf> = None;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--token-s" => {
                i += 1;
                token_s = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
                i += 1;
            }
            "--token-z" => {
                i += 1;
                token_z = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
                i += 1;
            }
            "--blocks" => {
                i += 1;
                blocks = Some(
                    args.get(i)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(|| usage()),
                );
                i += 1;
            }
            "--bond-type-feature" => {
                bond_type = true;
                i += 1;
            }
            "--partition" => {
                partition = true;
                i += 1;
            }
            "--reject-unused-file-keys" => {
                reject_unused = true;
                i += 1;
            }
            s if s.starts_with('-') => {
                eprintln!("unknown flag: {s}");
                usage();
            }
            _ => {
                if path.is_some() {
                    usage();
                }
                path = Some(PathBuf::from(&args[i]));
                i += 1;
            }
        }
    }

    let path = path.unwrap_or_else(|| usage());
    if !path.is_file() {
        eprintln!("not a file: {}", path.display());
        std::process::exit(2);
    }

    let device = boltr_backend_burn::default_device();
    let mut config = Boltz2BurnModelConfig::with_defaults(token_s, token_z, blocks);
    config.dims.bond_type_feature = bond_type;
    let model = Boltz2BurnModel::<NdArray>::new(&device, &config);

    let missing = match model.keys_missing_in_safetensors(path.as_path()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
    };
    let extra = match model.safetensors_keys_unused(path.as_path()) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
    };

    if partition {
        let names = match list_safetensor_names(path.as_path()) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("{e:#}");
                std::process::exit(1);
            }
        };
        let (infer, other) = partition_safetensors_keys_for_inference(&names);
        eprintln!(
            "Partition: {} inference-related keys, {} other keys",
            infer.len(),
            other.len()
        );
        eprintln!("Inference top-level prefixes: {BOLTZ2_INFERENCE_TOP_LEVEL_KEYS:?}");
    }

    let param_count = model.parameter_names().len();
    eprintln!("Burn skeleton parameters: {param_count}");
    eprintln!("Missing in file ({}):", missing.len());
    for k in missing.iter().take(30) {
        eprintln!("  {k}");
    }
    if missing.len() > 30 {
        eprintln!("  ... and {} more", missing.len() - 30);
    }
    eprintln!("Unused file keys vs skeleton ({}):", extra.len());
    for k in extra.iter().take(20) {
        eprintln!("  {k}");
    }
    if extra.len() > 20 {
        eprintln!("  ... and {} more", extra.len() - 20);
    }

    if !missing.is_empty() {
        eprintln!(
            "\nFAIL: {} skeleton keys absent from {}.",
            missing.len(),
            path.display()
        );
        std::process::exit(1);
    }

    if reject_unused && !extra.is_empty() {
        eprintln!(
            "\nFAIL: {} safetensors keys are not in the Burn skeleton registry.",
            extra.len()
        );
        std::process::exit(1);
    }

    eprintln!("OK: every Burn skeleton key is present in the file.");
    if !extra.is_empty() {
        eprintln!(
            "NOTE: {} checkpoint keys are not yet mapped (unported modules).",
            extra.len()
        );
    }
}
