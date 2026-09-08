#!/usr/bin/env bash
# Lock the RTX 3050 memory clock at its top state and report the link/clock state.
# On this laptop the driver otherwise parks the memory clock at 810 MHz (of 5501)
# under a 25 W software cap, which makes every memory-bound kernel about 7x slower.
# `nvidia-smi -pl` is not supported here; the memory clock lock is the practical fix
# and does not survive a reboot (install nvidia-memory-clock.service for that).
set -euo pipefail
case "${1:-status}" in
  lock)   nvidia-smi -lmc 5001,5501 ;;
  unlock) nvidia-smi -rmc ;;
  status) ;;
  *) echo "usage: $0 [lock|unlock|status]" >&2; exit 2 ;;
esac
nvidia-smi --query-gpu=name,pstate,clocks.mem,clocks.max.mem,clocks.sm,power.draw,power.limit,clocks_throttle_reasons.active --format=csv
dev=$(ls -d /sys/bus/pci/devices/*/ | while read -r d; do [ "$(cat "$d/class" 2>/dev/null)" = "0x030000" ] && [ "$(cat "$d/vendor" 2>/dev/null)" = "0x10de" ] && echo "$d" && break; done)
if [ -n "${dev:-}" ]; then
  echo "PCIe link: current $(cat "$dev/current_link_speed") x$(cat "$dev/current_link_width"), max $(cat "$dev/max_link_speed") x$(cat "$dev/max_link_width"); ASPM policy: $(cat /sys/module/pcie_aspm/parameters/policy 2>/dev/null)"
fi
