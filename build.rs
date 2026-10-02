use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    if std::env::var("PROFILE")? == "debug" {
        let out_dir = std::env::var("OUT_DIR")?;
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;

        Command::new("python3")
            .arg("-m")
            .arg("venv")
            .arg(".venv")
            .current_dir(PathBuf::from(&out_dir))
            .status()?;
        Command::new(".venv/bin/pip")
            .arg("install")
            .arg("-r")
            .arg(PathBuf::from(&manifest_dir).join("tests/requirements.txt"))
            .current_dir(PathBuf::from(&out_dir))
            .status()?;
        Command::new(".venv/bin/python")
            .arg(PathBuf::from(&manifest_dir).join("tests/generate.py"))
            .current_dir(PathBuf::from(&out_dir))
            .status()?;

        println!("cargo::rerun-if-changed=tests/requirements.txt");
        println!("cargo::rerun-if-changed=tests/generate.py");
        println!("cargo::rerun-if-changed=build.rs");
    }

    Ok(())
}
