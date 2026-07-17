#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UUID="pllm-foreground@local"
TARGET="${HOME}/.local/share/gnome-shell/extensions/${UUID}"

mkdir -p "${TARGET}"
cp "${ROOT_DIR}/desktop/gnome-extension/metadata.json" "${TARGET}/metadata.json"
cp "${ROOT_DIR}/desktop/gnome-extension/extension.js" "${TARGET}/extension.js"

if gnome-extensions list | grep -qx "${UUID}"; then
  gnome-extensions enable "${UUID}" || true
fi

echo "Installed ${UUID}. Log out and back in if GNOME has not loaded it yet."
