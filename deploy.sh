#!/bin/bash
set -e

DOMAIN="iiitd-chain.duckdns.org"
EMAIL="iiitd.solulab@gmail.com"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   IIITD Blockchain — Deploy Script     ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "  Domain: $DOMAIN"
echo ""

# ── Step 0: Ensure git + openssl are available ───────────
echo "── Checking prerequisites ──"
apt-get update -qq
apt-get install -y -qq git openssl > /dev/null 2>&1
echo "  ✅ git & openssl available"

# ── Step 1: Install Docker if not present ────────────────
if ! command -v docker &> /dev/null; then
    echo "── Installing Docker (official apt repo) ──"
    apt-get update
    apt-get install -y ca-certificates curl
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc

    echo "Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc" > /etc/apt/sources.list.d/docker.sources

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    systemctl enable docker
    systemctl start docker
    echo "  ✅ Docker installed (with compose plugin)"
else
    echo "  ✅ Docker already installed"
fi

# ── Step 1b: Open firewall ports if UFW is active ────────
if command -v ufw &> /dev/null && ufw status | grep -q "active"; then
    echo "── Opening firewall ports ──"
    ufw allow 80/tcp
    ufw allow 443/tcp
    echo "  ✅ Ports 80 & 443 opened"
fi

# ── Step 2: Add swap if RAM < 2GB (Rust compilation needs it) ──
echo ""
echo "── Checking memory ──"
TOTAL_RAM=$(free -m | awk '/^Mem:/{print $2}')
if [ "$TOTAL_RAM" -lt 2000 ]; then
    if [ ! -f /swapfile ]; then
        echo "  RAM is ${TOTAL_RAM}MB — adding 2GB swap for Rust build..."
        fallocate -l 2G /swapfile
        chmod 600 /swapfile
        mkswap /swapfile
        swapon /swapfile
        echo '/swapfile none swap sw 0 0' >> /etc/fstab
        echo "  ✅ 2GB swap added"
    else
        swapon /swapfile 2>/dev/null || true
        echo "  ✅ Swap already exists"
    fi
else
    echo "  ✅ RAM is ${TOTAL_RAM}MB — no swap needed"
fi

# ── Step 3: Create dummy certs so nginx can start ────────
echo ""
echo "── Setting up SSL ──"

CERT_PATH="./certbot/conf/live/$DOMAIN"
mkdir -p "$CERT_PATH"
mkdir -p "./certbot/www"

if [ ! -f "$CERT_PATH/fullchain.pem" ]; then
    echo "  Creating dummy certificate..."
    openssl req -x509 -nodes -newkey rsa:2048 -days 1 \
        -keyout "$CERT_PATH/privkey.pem" \
        -out "$CERT_PATH/fullchain.pem" \
        -subj "/CN=$DOMAIN" 2>/dev/null
    echo "  ✅ Dummy cert created"
else
    echo "  ✅ Cert already exists"
fi

# ── Step 4: Build and start containers ───────────────────
echo ""
echo "── Building & starting containers (Rust build takes a few minutes) ──"
docker compose up -d --build nginx backend
echo "  ✅ Backend + Nginx running"

# Wait for nginx to be ready
sleep 5

# ── Step 5: Get real SSL cert from Let's Encrypt ────────
echo ""
echo "── Requesting SSL certificate ──"

# Remove dummy cert
rm -rf "$CERT_PATH"

docker compose run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    -d "$DOMAIN"

# ── Step 6: Reload nginx with real cert ──────────────────
echo ""
echo "── Reloading Nginx ──"
docker compose restart nginx
echo "  ✅ Nginx restarted with real SSL"

# ── Step 7: Set up auto-renewal cron ─────────────────────
echo ""
echo "── Setting up cert auto-renewal ──"
CRON_JOB="0 3 * * * cd $(pwd) && docker compose run --rm certbot renew && docker compose restart nginx"
(crontab -l 2>/dev/null | grep -v "certbot renew"; echo "$CRON_JOB") | crontab -
echo "  ✅ Auto-renewal cron added (daily at 3am)"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✅ Deployment complete!                         ║"
echo "║                                                  ║"
echo "║  API:  https://$DOMAIN              ║"
echo "║  WS:   wss://$DOMAIN/ws             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Test it:  curl https://$DOMAIN/status"
echo ""
