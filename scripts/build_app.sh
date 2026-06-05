#!/usr/bin/env bash
set -euo pipefail

# Build a signed, notarized Subtitle Anywhere.app and DMG.
#
# Usage:
#   ./scripts/build_app.sh              # build only (no signing)
#   ./scripts/build_app.sh --sign       # build + sign + notarize + DMG
#
# Required env vars for --sign:
#   ASC_KEY_P8_PATH  — path to App Store Connect .p8 key file
#   ASC_KEY_ID       — App Store Connect key ID
#   ASC_ISSUER_ID    — App Store Connect issuer ID
#
# Requires: cargo, uv, rsync (+ codesign, notarytool, hdiutil for --sign)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
APP_NAME="Subtitle Anywhere"
BUNDLE_ID="io.github.madeye.subtitle-anywhere"
VERSION="0.1.0"
APP_BUNDLE="$BUILD_DIR/${APP_NAME}.app"
CONTENTS="$APP_BUNDLE/Contents"
PYTHON_VERSION="3.13"
ENTITLEMENTS="$REPO_ROOT/macos/entitlements.plist"

SIGN=false
if [[ "${1:-}" == "--sign" ]]; then
    SIGN=true
    : "${ASC_KEY_P8_PATH:?Set ASC_KEY_P8_PATH to your App Store Connect .p8 key file}"
    : "${ASC_KEY_ID:?Set ASC_KEY_ID to your App Store Connect key ID}"
    : "${ASC_ISSUER_ID:?Set ASC_ISSUER_ID to your App Store Connect issuer ID}"
fi

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

UV_PYTHON_BASE="$HOME/.local/share/uv/python"
UV_PYTHON_DIR=$(ls -d "$UV_PYTHON_BASE"/cpython-${PYTHON_VERSION}*-macos-aarch64-none 2>/dev/null | sort -V | tail -1)
if [ -z "$UV_PYTHON_DIR" ] || [ ! -d "$UV_PYTHON_DIR" ]; then
    echo "ERROR: standalone Python $PYTHON_VERSION not found in $UV_PYTHON_BASE"
    exit 1
fi
echo "    Standalone Python: $UV_PYTHON_DIR"

PYTHON_DIR="$CONTENTS/Resources/python"
echo "==> Copying Python runtime into bundle"
rsync -a "$UV_PYTHON_DIR/" "$PYTHON_DIR/"

PYTHON_BIN=$(ls "$PYTHON_DIR/bin/python${PYTHON_VERSION}"* 2>/dev/null | grep -v config | head -1)
echo "    Bundled Python: $PYTHON_BIN"

# --- 3. Install lightweight dependencies (ML deps installed on first launch) ---
echo "==> Installing base Python dependencies"
find "$PYTHON_DIR" -name EXTERNALLY-MANAGED -delete 2>/dev/null || true
uv pip install --python "$PYTHON_BIN" --break-system-packages \
    numpy huggingface_hub 2>&1

# --- 4. Copy application code ---
echo "==> Copying application code"
APP_DIR="$CONTENTS/Resources/app"
mkdir -p "$APP_DIR"
rsync -a --exclude='__pycache__' "$REPO_ROOT/server/" "$APP_DIR/server/"
rsync -a "$REPO_ROOT/web/" "$APP_DIR/web/"

# --- 5. Create Info.plist ---
echo "==> Writing Info.plist"
cat > "$CONTENTS/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
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

if [ "$SIGN" = false ]; then
    APP_SIZE=$(du -sh "$APP_BUNDLE" | cut -f1)
    echo ""
    echo "==> Build complete (unsigned): $APP_BUNDLE ($APP_SIZE)"
    echo "    Run with: open '$APP_BUNDLE'"
    echo "    To sign:  $0 --sign"
    exit 0
fi

# === Signing & Notarization ===

IDENTITY=$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 | awk -F'"' '{print $2}')
if [ -z "$IDENTITY" ]; then
    echo "ERROR: No 'Developer ID Application' certificate found in keychain"
    exit 1
fi
echo "==> Signing identity: $IDENTITY"

# --- 6. Deep-sign the bundle ---
echo "==> Signing app bundle"

# Sign all nested binaries and dylibs first (Python, .so files, etc.)
find "$CONTENTS/Resources" -type f \( -name '*.so' -o -name '*.dylib' \) -print0 | \
    xargs -0 -n1 codesign --force --sign "$IDENTITY" --timestamp \
        --options runtime --entitlements "$ENTITLEMENTS" 2>&1

codesign --force --sign "$IDENTITY" --timestamp \
    --options runtime --entitlements "$ENTITLEMENTS" \
    "$PYTHON_BIN" 2>&1

# Sign the main executable and bundle
codesign --force --sign "$IDENTITY" --timestamp \
    --options runtime --entitlements "$ENTITLEMENTS" \
    --deep "$APP_BUNDLE" 2>&1

codesign --verify --deep --strict "$APP_BUNDLE"
echo "    Signature OK"

# --- 7. Notarize the app ---
echo "==> Notarizing app"
NOTARIZE_ZIP="/tmp/${APP_NAME}-notarize.zip"
rm -f "$NOTARIZE_ZIP"
ditto -c -k --keepParent "$APP_BUNDLE" "$NOTARIZE_ZIP"

xcrun notarytool submit "$NOTARIZE_ZIP" \
    --key "$ASC_KEY_P8_PATH" --key-id "$ASC_KEY_ID" --issuer "$ASC_ISSUER_ID" \
    --wait

echo "==> Stapling app"
xcrun stapler staple "$APP_BUNDLE"
rm -f "$NOTARIZE_ZIP"

# --- 8. Create DMG ---
echo "==> Creating DMG"
DMG_NAME="${APP_NAME}-${VERSION}"
DMG_PATH="$BUILD_DIR/${DMG_NAME}.dmg"
DMG_DIR="/tmp/${APP_NAME}-dmg"
RW_DMG="/tmp/${DMG_NAME}-rw.dmg"
rm -rf "$DMG_DIR" "$DMG_PATH" "$RW_DMG"
mkdir -p "$DMG_DIR"

cp -R "$APP_BUNDLE" "$DMG_DIR/"
ln -s /Applications "$DMG_DIR/Applications"

hdiutil create -volname "$APP_NAME" -srcfolder "$DMG_DIR" -ov -format UDRW "$RW_DMG"
rm -rf "$DMG_DIR"

# Customize DMG Finder layout
MOUNT_POINT=$(hdiutil attach -readwrite -noverify "$RW_DMG" | grep "/Volumes/" | tail -1 | awk -F'\t' '{print $NF}')
echo "    Mounted at: $MOUNT_POINT"

osascript <<APPLESCRIPT
tell application "Finder"
    tell disk "$APP_NAME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {100, 100, 640, 400}
        set viewOptions to the icon view options of container window
        set arrangement of viewOptions to not arranged
        set icon size of viewOptions to 128
        set position of item "${APP_NAME}.app" of container window to {120, 150}
        set position of item "Applications" of container window to {420, 150}
        close
        open
        update without registering applications
        delay 1
        close
    end tell
end tell
APPLESCRIPT

sync
hdiutil detach "$MOUNT_POINT"

hdiutil convert "$RW_DMG" -format UDZO -imagekey zlib-level=9 -o "$DMG_PATH"
rm -f "$RW_DMG"

# --- 9. Sign + notarize DMG ---
echo "==> Signing DMG"
codesign --sign "$IDENTITY" --timestamp "$DMG_PATH"

echo "==> Notarizing DMG"
xcrun notarytool submit "$DMG_PATH" \
    --key "$ASC_KEY_P8_PATH" --key-id "$ASC_KEY_ID" --issuer "$ASC_ISSUER_ID" \
    --wait

echo "==> Stapling DMG"
xcrun stapler staple "$DMG_PATH"

DMG_SIZE=$(du -h "$DMG_PATH" | cut -f1)
APP_SIZE=$(du -sh "$APP_BUNDLE" | cut -f1)
echo ""
echo "=== Done ==="
echo "App: $APP_BUNDLE ($APP_SIZE)"
echo "DMG: $DMG_PATH ($DMG_SIZE)"
echo "Version: $VERSION"
