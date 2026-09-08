#!/usr/bin/env bash
# Test and raise the PCIe link speed of the NVIDIA GPU. Run as root.
#
#   sudo deploy/doomfly/pcie-link.sh status    # registers, advertised/negotiated speed, loaded transfer rate
#   sudo deploy/doomfly/pcie-link.sh try       # apply the reversible steps one at a time, measuring after each
#   sudo deploy/doomfly/pcie-link.sh restore   # undo everything `try` changed (clocks, target speed)
#   sudo deploy/doomfly/pcie-link.sh persist   # write the modprobe options for the next boot (reversible: remove the file)
#
# Background (RTX 3050 Laptop, driver 595): both ends support gen 4, but the card advertises
# gen 1 in its Link Capabilities while the driver's power management is active, so the link
# trains at 2.5 GT/s and host<->device copies run at ~1.6 GB/s. This script sets the root port
# target speed, retrains, locks GPU clocks, and measures under a real transfer load. Nothing here
# survives a reboot except `persist`.
set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
PY="$ROOT/.venv-neural/bin/python"
MODPROBE_CONF=/etc/modprobe.d/nvidia-pm.conf

[ "$(id -u)" = 0 ] || { echo "run as root (sudo)" >&2; exit 1; }

gpu_bdf() {
  for d in /sys/bus/pci/devices/*; do
    if [ "$(cat "$d/vendor" 2>/dev/null)" = "0x10de" ] && [ "$(cat "$d/class" 2>/dev/null)" = "0x030000" ]; then basename "$d"; return; fi
  done
  echo "no NVIDIA GPU found" >&2; exit 1
}
GPU=$(gpu_bdf)
PORT=$(basename "$(readlink -f /sys/bus/pci/devices/$GPU/..)")
SYS=/sys/bus/pci/devices/$GPU

speed_now() { cat "$SYS/current_link_speed"; }

registers() {
  echo "GPU $GPU behind root port $PORT"
  echo "  sysfs: current $(speed_now) x$(cat $SYS/current_link_width), max $(cat $SYS/max_link_speed) x$(cat $SYS/max_link_width)"
  echo "  root port: current $(cat /sys/bus/pci/devices/$PORT/current_link_speed), max $(cat /sys/bus/pci/devices/$PORT/max_link_speed)"
  echo "  target speed (Link Control 2, low nibble; 1=2.5 2=5 3=8 4=16 GT/s): port=$(setpci -s $PORT CAP_EXP+30.w) gpu=$(setpci -s $GPU CAP_EXP+30.w)"
  lspci -vv -s "$GPU" | grep -E 'LnkCap:|LnkSta:|LnkCtl:' | sed 's/^/  /'
  nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem,pcie.link.gen.gpucurrent,pcie.link.gen.max,pcie.link.gen.gpumax,pcie.link.gen.hostmax --format=csv,noheader | sed 's/^/  nvidia-smi: pstate, sm, mem, gen current, gen max negotiated, gpu max, host max = /'
}

transfer_test() {
  # Pinned host->device copies of 64 MB; prints GB/s and the link speed sampled during the copies.
  [ -x "$PY" ] || { echo "  (no venv python at $PY; skipping transfer test)"; return; }
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 CUPY_CACHE_DIR=/tmp/cupy-root "$PY" - "$SYS" <<'PY' 2>/dev/null || echo "  transfer test failed (cupy import?)"
import sys, time, numpy as np, cupy as cp
sysdir = sys.argv[1]
m = cp.cuda.alloc_pinned_memory(64 << 20); h = np.frombuffer(m, np.uint8, 64 << 20); d = cp.zeros(64 << 20, cp.uint8)
d.set(h); cp.cuda.Stream.null.synchronize()
speeds = set(); t = time.perf_counter()
for i in range(30):
    d.set(h)
    if i % 5 == 0: speeds.add(open(sysdir + '/current_link_speed').read().strip())
cp.cuda.Stream.null.synchronize(); dt = (time.perf_counter() - t) / 30
print('  transfer: pinned H2D %.1f GB/s, link during copies: %s' % (64 / 1024 / dt, ', '.join(sorted(speeds))))
PY
}

retrain() {
  # target 16 GT/s on the root port (and the GPU, whose field is normally 0 = "use max"), then retrain
  setpci -s "$PORT" CAP_EXP+30.w=0x0004:0x000f
  setpci -s "$GPU"  CAP_EXP+30.w=0x0004:0x000f
  setpci -s "$PORT" CAP_EXP+10.w=0x0020:0x0020
  sleep 0.5
}

case "${1:-status}" in
  status)
    registers; transfer_test ;;
  try)
    echo "== before"; registers; transfer_test
    echo "== step 1: root-port target speed 16 GT/s + retrain"; retrain; registers | grep -E 'sysfs|target'; transfer_test
    echo "== step 2: lock GPU clocks (nvidia-smi -lgc / -lmc) + retrain"
    nvidia-smi -lmc 5001,5501 >/dev/null; nvidia-smi -lgc 2100,2100 >/dev/null; sleep 1; retrain; registers | grep -E 'sysfs|LnkCap|nvidia-smi'; transfer_test
    echo "== step 3: kernel PCIe power policy = performance (runtime PM on for GPU and port) + retrain"
    [ -w /sys/module/pcie_aspm/parameters/policy ] && echo performance > /sys/module/pcie_aspm/parameters/policy
    echo on > "$SYS/power/control"; echo on > "/sys/bus/pci/devices/$PORT/power/control"; sleep 1; retrain; registers | grep -E 'sysfs|LnkCap'; transfer_test
    echo "== result: link now $(speed_now). If still 2.5 GT/s the card's firmware caps it in this power mode:"
    echo "   run '$0 persist' and reboot (driver runtime PM off, max performance policy), or change the BIOS/graphics mode."
    echo "   '$0 restore' undoes the clock locks and runtime PM changes." ;;
  restore)
    nvidia-smi -rgc >/dev/null; echo auto > "$SYS/power/control"; echo auto > "/sys/bus/pci/devices/$PORT/power/control"
    [ -w /sys/module/pcie_aspm/parameters/policy ] && echo default > /sys/module/pcie_aspm/parameters/policy
    setpci -s "$PORT" CAP_EXP+30.w=0x0001:0x000f; setpci -s "$GPU" CAP_EXP+30.w=0x0000:0x000f; setpci -s "$PORT" CAP_EXP+10.w=0x0020:0x0020
    echo "restored (memory clock lock kept; 'nvidia-smi -rmc' releases it)"; registers | grep -E 'sysfs|target' ;;
  persist)
    printf 'options nvidia NVreg_DynamicPowerManagement=0x00 NVreg_RegistryDwords="PerfLevelSrc=0x2222;OverrideMaxPerf=0x1"\n' > "$MODPROBE_CONF"
    update-initramfs -u
    echo "wrote $MODPROBE_CONF; reboot, re-lock the memory clock (gpu-clocks.sh lock), then run '$0 try'."
    echo "undo: rm $MODPROBE_CONF && update-initramfs -u" ;;
  *) echo "usage: $0 [status|try|restore|persist]" >&2; exit 2 ;;
esac
