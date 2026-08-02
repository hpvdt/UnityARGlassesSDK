// Copyright (C) 2023, Alex Badics
// This file is part of ar-drivers-rs
// Licensed under the MIT license. See LICENSE file in the project root for details.

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ar_drivers::{xreal_air::XrealAir, ARGlasses};
use clap::Parser;

#[derive(Parser)]
#[command(about = "Capture a timed XREAL Air 1 raw IMU packet trace")]
struct Args {
    /// Destination for the versioned packet log.
    #[arg(long, default_value = "tests/fixtures/xreal_air_air1_60s.log")]
    output: PathBuf,

    /// Capture duration in seconds.
    #[arg(long, default_value_t = 60)]
    seconds: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let mut glasses = XrealAir::new()?;

    if glasses.name() != "XREAL Air" {
        return Err(format!(
            "this capture fixture requires an XREAL Air 1, but found {}",
            glasses.name()
        )
        .into());
    }

    if let Some(parent) = args
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    let duration = Duration::from_secs(args.seconds);
    let start = Instant::now();
    let mut last_instruction = None;

    println!(
        "Capturing raw Air 1 IMU packets to {}",
        args.output.display()
    );
    println!("Begin the continuous calibration dance now.");
    glasses.start_packet_logging(&args.output)?;

    let capture_result = (|| {
        while start.elapsed() < duration {
            let elapsed_seconds = start.elapsed().as_secs();
            let instruction_index = (elapsed_seconds / 5) % 3;
            if last_instruction != Some(instruction_index) {
                let instruction = match instruction_index {
                    0 => "Smoothly roll the glasses through wide left/right rotations.",
                    1 => "Smoothly pitch the glasses through wide up/down rotations.",
                    _ => "Smoothly yaw the glasses through wide horizontal rotations.",
                };
                println!(
                    "[{elapsed_seconds:02}/{:02}s] {instruction} Keep all motion continuous.",
                    args.seconds
                );
                last_instruction = Some(instruction_index);
            }
            glasses.read_event()?;
        }
        Ok::<(), ar_drivers::Error>(())
    })();
    let stop_result = glasses.stop_packet_logging();

    capture_result?;
    stop_result?;
    println!("Capture complete. You can stop the calibration dance.");
    Ok(())
}
