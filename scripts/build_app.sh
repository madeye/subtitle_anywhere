#!/usr/bin/env bash
set -euo pipefail

# Build Subtitle Anywhere.app — a native macOS app with Rust webview,
# bundled standalone Python runtime, and all dependencies.
#
# Usage: ./scripts/build_app.sh
#
# Requires: cargo, uv, rsync

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
APP_NAME="Subtitle Anywhere"
APP_BUNDLE="$BUILD_DIR/${APP_NAME}.app"
CONTENTS="$APP_BUNDLE/Contents"
PYTHON_VERSION="3.13"

echo "==> Cleaning previous build"
rm -rf "$APP_BUNDLE"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

# --- 1. Build the Rust binary ---
echo "==> Building Rust binary (release)"
cd "$REPO_ROOT/macos"
cargo build --release 2>&1
cp target/release/subtitle-anywhere "$CONTENTS/MacOS/subtitle-anywhere"

# --- 2. Bundle standalone Python runtime ---
echo "==> Installing standalone Python $PYTHON_VERSION via uv"
uv python install "$PYTHON_VERSION" 2>&1

# Find the uv-managed standalone Python installation
UV_PYTHON_BASE="$HOME/.local/share/uv/python"
UV_PYTHON_DIR=$(ls -d "$UV_PYTHON_BASE"/cpython-${PYTHON_VERSION}*-macos-aarch64-none 2>/dev/null | sort -V | tail -1)
if [ -z "$UV_PYTHON_DIR" ] || [ ! -d "$UV_PYTHON_DIR" ]; then
    echo "ERROR: standalone Python $PYTHON_VERSION not found in $UV_PYTHON_BASE"
    exit 1
fi
echo "    Standalone Python: $UV_PYTHON_DIR"

# Copy the full standalone Python into the bundle
PYTHON_DIR="$CONTENTS/Resources/python"
echo "==> Copying Python runtime into bundle"
rsync -a "$UV_PYTHON_DIR/" "$PYTHON_DIR/"

PYTHON_BIN=$(ls "$PYTHON_DIR/bin/python${PYTHON_VERSION}"* 2>/dev/null | grep -v config | head -1)
echo "    Bundled Python: $PYTHON_BIN"

# --- 3. Install dependencies into the bundled Python ---
echo "==> Installing Python dependencies"
# Remove the externally-managed marker so we can install into this copy
find "$PYTHON_DIR" -name EXTERNALLY-MANAGED -delete 2>/dev/null || true
uv pip install --python "$PYTHON_BIN" --break-system-packages \
    numpy huggingface_hub mlx-whisper mlx-lm 2>&1

# --- 4. Copy application code ---
echo "==> Copying application code"
APP_DIR="$CONTENTS/Resources/app"
mkdir -p "$APP_DIR"
rsync -a --exclude='__pycache__' "$REPO_ROOT/server/" "$APP_DIR/server/"
rsync -a "$REPO_ROOT/web/" "$APP_DIR/web/"

# --- 5. Create Info.plist ---
echo "==> Writing Info.plist"
cat > "$CONTENTS/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Subtitle Anywhere</string>
    <key>CFBundleDisplayName</key>
    <string>Subtitle Anywhere</string>
    <key>CFBundleIdentifier</key>
    <string>io.github.madeye.subtitle-anywhere</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundleExecutable</key>
    <string>subtitle-anywhere</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
</dict>
</plist>
PLIST

# --- 6. Summary ---
APP_SIZE=$(du -sh "$APP_BUNDLE" | cut -f1)
echo ""
echo "==> Build complete: $APP_BUNDLE ($APP_SIZE)"
echo "    Run with: open '$APP_BUNDLE'"
