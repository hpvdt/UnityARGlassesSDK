#!/usr/bin/env bash

CRATE_DIR="$(cd "$(dirname "$0")/../__module~/driver"; pwd)"
cd "$CRATE_DIR"

rustup target add x86_64-pc-windows-gnu aarch64-apple-darwin x86_64-unknown-linux-gnu

cargo clean

echo "[Start Compiling ... Zig compiler and cargo-zigbuild must be installed]"

# TODO: disable until https://github.com/rust-cross/cargo-zigbuild/issues/251 is fixed
cargo zigbuild --target x86_64-pc-windows-gnu --release || echo "WARNING: Failed to build for x86_64-pc-windows-gnu"

cargo zigbuild --target aarch64-apple-darwin --release || echo "WARNING: Failed to build for aarch64-apple-darwin"

# TODO: this will fail on OS without libudev (I mean linux)
cargo zigbuild --target x86_64-unknown-linux-gnu --release || \
  cargo zigbuild --target x86_64-unknown-linux-gnu --release --no-default-features --features rokid,xreal,grawoow

echo "[Copying to Plugins ...]"
[ -f target/x86_64-pc-windows-gnu/release/ar_drivers.dll ] && cp target/x86_64-pc-windows-gnu/release/ar_drivers.dll ../../Plugins/
[ -f target/aarch64-apple-darwin/release/libar_drivers.dylib ] && cp target/aarch64-apple-darwin/release/libar_drivers.dylib ../../Plugins/
cp target/x86_64-unknown-linux-gnu/release/libar_drivers.so ../../Plugins/

