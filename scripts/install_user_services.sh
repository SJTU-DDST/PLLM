#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${HOME}/.config/systemd/user"
CURRENT_USER="${USER:-$(id -un)}"
mkdir -p "${TARGET}"

sed "s|@ROOT_DIR@|${ROOT_DIR}|g; s|@HOME@|${HOME}|g; s|@USER@|${CURRENT_USER}|g" \
  "${ROOT_DIR}/systemd/pllm-daemon.service" > "${TARGET}/pllm-daemon.service"
sed "s|@ROOT_DIR@|${ROOT_DIR}|g; s|@HOME@|${HOME}|g; s|@USER@|${CURRENT_USER}|g" \
  "${ROOT_DIR}/systemd/pllm-desktop.service" > "${TARGET}/pllm-desktop.service"
sed "s|@ROOT_DIR@|${ROOT_DIR}|g; s|@HOME@|${HOME}|g; s|@USER@|${CURRENT_USER}|g" \
  "${ROOT_DIR}/systemd/pllm-rdma-store.service" > "${TARGET}/pllm-rdma-store.service"
sed "s|@ROOT_DIR@|${ROOT_DIR}|g; s|@HOME@|${HOME}|g; s|@USER@|${CURRENT_USER}|g" \
  "${ROOT_DIR}/systemd/pllm-rdma-state.service" > "${TARGET}/pllm-rdma-state.service"

systemctl --user daemon-reload
systemctl --user enable pllm-daemon.service pllm-desktop.service
echo "Installed PLLM services. RDMA tiers remain opt-in: systemctl --user enable --now pllm-rdma-store pllm-rdma-state"
