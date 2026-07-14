#!/usr/bin/env bash
# Run ONE arm of the CosyVoice2-vs-3 A/B under real MuseTalk render contention.
#   ab_arm.sh <v2|v3> <cycle>
# Enforces P15 load order: CosyVoice (vLLM) loads on a near-empty card, THEN MuseTalk.
# MuseTalk's workload is held identical across arms by looping one fixed WAV through it.
set -u
ARM="$1"; CYCLE="$2"
VIS=/e/Claude/visualllm
COSY=/e/Claude/VisualLLm/tts/cosyvoice-server
WSL_IP=172.24.44.238
MUSETALK_PY='E:\miniconda3\envs\musetalk\python.exe'

kill_port() {  # $1 = port ; native TerminateProcess (taskkill hangs under load on this box)
  python - "$1" <<'PY'
import ctypes, subprocess, sys
port = sys.argv[1]
out = subprocess.run(["netstat","-ano"], capture_output=True, text=True).stdout
pids = {l.split()[-1] for l in out.splitlines() if f":{port} " in l and "LISTENING" in l}
for p in pids:
    h = ctypes.windll.kernel32.OpenProcess(0x0001, False, int(p))
    if h:
        ctypes.windll.kernel32.TerminateProcess(h, 0); ctypes.windll.kernel32.CloseHandle(h)
        print(f"killed pid {p} on :{port}")
PY
}

echo "=== [$ARM cycle$CYCLE] tearing down ==="
pkill -f _drive_frames 2>/dev/null
kill_port 8002
wsl -d Ubuntu -e bash -c "pkill -9 -f uvicorn 2>/dev/null" >/dev/null 2>&1
sleep 4

echo "=== [$ARM] starting CosyVoice on a near-empty card (P15) ==="
if [ "$ARM" = "v3" ]; then
  wsl -d Ubuntu -e bash -c '
    export COSYVOICE_MODEL_DIR=/mnt/e/Claude/VisualLLm/tts/cosyvoice-server/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512
    export COSYVOICE_PROMPT_TEXT="You are a helpful assistant.<|endofprompt|>你好，我是你的AI虚拟助手，很高兴见到你。今天天气不错，有什么我可以帮你的"
    bash /mnt/e/Claude/VisualLLm/tts/cosyvoice-server/run_vllm_server.sh' > "$COSY/logs/ab_${ARM}_c${CYCLE}.log" 2>&1 &
else
  wsl -d Ubuntu -e bash -c 'bash /mnt/e/Claude/VisualLLm/tts/cosyvoice-server/run_vllm_server.sh' \
    > "$COSY/logs/ab_${ARM}_c${CYCLE}.log" 2>&1 &
fi
for i in $(seq 1 72); do
  curl -s -m 2 "http://$WSL_IP:8001/health" >/dev/null 2>&1 && { echo "  cosyvoice healthy (~$((i*5))s)"; break; }
  sleep 5
done
curl -s -m 2 "http://$WSL_IP:8001/health" >/dev/null 2>&1 || { echo "FATAL: cosyvoice never came up"; exit 1; }

echo "=== [$ARM] starting MuseTalk (after cosyvoice) ==="
cd "$VIS"
set -a; . ./.env 2>/dev/null; set +a
"$MUSETALK_PY" -u -m local_services.musetalk_server.app > "$VIS/output/ab_musetalk_${ARM}_c${CYCLE}.log" 2>&1 &
for i in $(seq 1 60); do
  (netstat -ano | grep -q ":8002 .*LISTENING") && { echo "  musetalk up (~$((i*3))s)"; break; }
  sleep 3
done
netstat -ano | grep -q ":8002 .*LISTENING" || { echo "FATAL: musetalk never bound :8002"; exit 1; }

echo "=== [$ARM] holding render contention (fixed WAV loop) ==="
( while true; do python -m scripts._drive_frames output/reply_concise.wav 12 >/dev/null 2>&1 || sleep 1; done ) &
DRIVE=$!
sleep 8   # let the renderer reach steady state before probing

echo "=== [$ARM cycle$CYCLE] probing ==="
cd "$COSY" && python _ab_run.py --host "$WSL_IP" --tag "$ARM" --cycle "$CYCLE" 2>&1 | tail -6
RC=$?

kill $DRIVE 2>/dev/null; pkill -f _drive_frames 2>/dev/null
echo "=== [$ARM cycle$CYCLE] done (rc=$RC) ==="
exit $RC
