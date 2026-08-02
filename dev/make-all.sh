#!/usr/bin/env bash

set -euo pipefail

CRATE_DIR="$(cd "$(dirname "$0")/../__module~/driver"; pwd)"
PLUGINS_DIR="../../Plugins"

cd "$CRATE_DIR"

#cargo clean

echo "[Start Compiling]"
cargo build --release

# Determine the library filename based on the target OS
case "$(uname -s)" in
    Linux)  LIB="libar_drivers.so"  ;;
    Darwin) LIB="libar_drivers.dylib" ;;
    *)      LIB="ar_drivers.dll"    ;;
esac

# If cross-compiling, output is in a target/<triple>/ subdirectory
TARGET_DIR="target"
if [ -n "${CARGO_BUILD_TARGET:-}" ]; then
    TARGET_DIR="target/$CARGO_BUILD_TARGET"
fi

cp "$TARGET_DIR/release/$LIB" "$PLUGINS_DIR"
