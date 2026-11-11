#!/usr/bin/env bash
# One-time setup of a Debian 13 server, run as root: setup-server.sh [--tunnel | domain]
# Without a domain the app runs on 127.0.0.1:8000 only; rerun with the domain
# once its A record points here, and Caddy serves it over HTTPS.
# With --tunnel no port is opened: Cloudflare reaches Caddy on 127.0.0.1:8080
# through cloudflared. It needs `cloudflared tunnel login` on the server first;
# a domain moves to the tunnel only on
# `cloudflared tunnel route dns --overwrite-dns sudoku-solver <domain>`.
set -euo pipefail
TUNNEL=
[ "${1:-}" = --tunnel ] && { TUNNEL=1; shift; }
DOMAIN=${1:-}
HERE=$(cd "$(dirname "$0")" && pwd)

apt-get update -q
# libgl1/libglib2.0: opencv-python's shared libraries; caddy: TLS + reverse proxy
apt-get install -y -q python3-venv libgl1 libglib2.0-0t64 caddy sqlite3

# 2 GB RAM and no swap: a 1 GB swap file keeps a spike from OOM-killing the app.
if ! swapon --show | grep -q /swapfile; then
    fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

id sudoku >/dev/null 2>&1 || useradd --system --home-dir /var/lib/sudoku-solver --shell /usr/sbin/nologin sudoku
install -d -o sudoku -g sudoku -m 750 /var/lib/sudoku-solver
install -d /opt/sudoku-solver/releases
[ -x /opt/sudoku-solver/venv/bin/python ] || python3 -m venv /opt/sudoku-solver/venv

install -m 644 "$HERE/sudoku-solver.service" /etc/systemd/system/sudoku-solver.service
systemctl daemon-reload
systemctl enable sudoku-solver

if [ -n "$TUNNEL" ]; then
    if ! command -v cloudflared >/dev/null; then
        curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg -o /usr/share/keyrings/cloudflare-main.gpg
        echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' \
            > /etc/apt/sources.list.d/cloudflared.list
        apt-get update -q && apt-get install -y -q cloudflared
    fi
    install -m 644 "$HERE/Caddyfile.tunnel" /etc/caddy/Caddyfile
    systemctl enable caddy
    systemctl restart caddy
    CRED=/etc/cloudflared/sudoku-solver.json
    if [ ! -f $CRED ]; then
        [ -f ~/.cloudflared/cert.pem ] \
            || { echo "Run 'cloudflared tunnel login' first, then rerun." >&2; exit 1; }
        install -d -m 700 /etc/cloudflared
        cloudflared tunnel create --credentials-file $CRED sudoku-solver
    fi
    TUNNEL_ID=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["TunnelID"])' $CRED)
    sed "s/TUNNEL_ID/$TUNNEL_ID/" "$HERE/cloudflared.yml" > /etc/cloudflared/config.yml
    systemctl is-enabled cloudflared >/dev/null 2>&1 || cloudflared service install
    systemctl restart cloudflared
    echo "Server ready; now run deploy/deploy.sh from a checkout, then route the domain to the tunnel."
    exit 0
fi
if [ -z "$DOMAIN" ]; then
    systemctl disable --now caddy
    echo "Server ready (no domain: Caddy disabled); now run deploy/deploy.sh from a checkout."
    exit 0
fi
install -m 644 "$HERE/Caddyfile" /etc/caddy/Caddyfile
mkdir -p /etc/systemd/system/caddy.service.d
printf '[Service]\nEnvironment=SUDOKU_DOMAIN=%s\n' "$DOMAIN" > /etc/systemd/system/caddy.service.d/domain.conf
systemctl daemon-reload
systemctl enable caddy
systemctl restart caddy
echo "Server ready; now run deploy/deploy.sh from a checkout."
