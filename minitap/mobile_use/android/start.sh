#!/data/data/com.termux/files/usr/bin/bash
# Start the mobile-use Android UI as a background service inside Termux.
#
#   ./start.sh            # background, survives closing the terminal
#   ./start.sh --fg       # stay in the foreground
#
# Environment (all optional):
#   ADB_HOST / ADB_PORT   adb server to reach; defaults to the loopback adbd
#   MOBILE_USE_ANDROID_PORT   listen port (default 8770)
set -u

PREFIX=/data/data/com.termux/files/usr
HOME_DIR=/data/data/com.termux/files/home
APP_DIR="$HOME_DIR/mobile-use"

export PATH="$PREFIX/bin:/system/bin"
export LD_LIBRARY_PATH="$PREFIX/lib"
export TMPDIR="$PREFIX/tmp"
export PYTHONPATH="$APP_DIR"
export PYTHONUNBUFFERED=1
# Telemetry is noise for a phone-hosted agent.
export MOBILE_USE_TELEMETRY_ENABLED=false

mkdir -p "$TMPDIR" "$HOME_DIR/logs"

cd "$APP_DIR" || { echo "app dir missing: $APP_DIR"; exit 1; }

# Single place for config: the same .env the agent reads for API keys also
# carries ADB_HOST / ADB_PORT / MOBILE_USE_DEVICE_ID. This has to happen
# *before* the derived exports below, or they fall back to the loopback adbd.
if [ -f "$APP_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$APP_DIR/.env"
  set +a
fi

# adbutils reads these to build its *default* client, and that default client is
# what uiautomator2 uses internally (u2.connect() takes only a serial). Setting
# them here is what lets the on-device agent reach the adb server -- without a
# working `adb` binary anywhere on the phone, which is the whole point here.
export ANDROID_ADB_SERVER_HOST="${ADB_HOST:-127.0.0.1}"
export ANDROID_ADB_SERVER_PORT="${ADB_PORT:-5037}"
[ -n "${MOBILE_USE_DEVICE_ID:-}" ] && export ANDROID_SERIAL="$MOBILE_USE_DEVICE_ID"

PORT="${MOBILE_USE_ANDROID_PORT:-8770}"
LOG="$HOME_DIR/logs/mobile-use.log"
PIDF="$HOME_DIR/logs/mobile-use.pid"

if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "already running (pid $(cat "$PIDF")) on port $PORT"
  exit 0
fi

# Hold a wakelock so Android does not freeze the agent mid-run when the screen
# goes off. Without this the process gets culled after a few idle minutes.
"$PREFIX/bin/termux-wake-lock" 2>/dev/null || true

start_bg() {
  nohup setsid "$PREFIX/bin/python" -m minitap.mobile_use.android \
      >>"$LOG" 2>&1 </dev/null &
  echo $! > "$PIDF"
  sleep 2
  # Termux's `hostname` has no -I, and `ip` cannot bind a netlink socket under
  # the app sandbox. Ask the kernel which source address it would route via.
  lan_ip="$("$PREFIX/bin/python" - <<'PY' 2>/dev/null
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("192.0.2.1", 80))  # TEST-NET-1: never routed, no packet sent
    print(s.getsockname()[0])
except Exception:
    pass
finally:
    s.close()
PY
)"
  echo "started pid $(cat "$PIDF")  log $LOG"
  echo "  on the phone: http://127.0.0.1:$PORT"
  [ -n "$lan_ip" ] && echo "  from a laptop: http://$lan_ip:$PORT"
}

if [ "${1:-}" = "--fg" ]; then
  exec "$PREFIX/bin/python" -m minitap.mobile_use.android
else
  start_bg
fi
