# llama@ — hardened systemd user units

Artifacts in this directory are **static text**. Nothing here has been
installed or enabled. Commands below are the exact steps to adopt them.

## Files

| File | Purpose |
|---|---|
| `llama@.service` | Hardened template unit (Type=simple — b9139 has no sd_notify: verified no `libsystemd` in `ldd`, no `sd_notify` symbol) |
| `../env/<name>.env` | One env file per registry model (`MODEL_FILE`, `PORT`, `ARGS` with full v2 flags) |
| `llama-health.sh` | Watchdog; restarts `llama@<name>` after 2 consecutive dead checks on bot-contract ports 8080/8081, never while a model is loading |

Hardening summary: `Restart=on-failure`, `RestartSec=10`,
`StartLimitBurst=5` (per 300 s), `LimitNOFILE=65535`,
`Environment=LD_PRELOAD=` + `CUDA_VISIBLE_DEVICES=0` (defeats the gpu-guard
shim per FACTS.md), sleep-free `ExecStartPre` port-free health-guard
(polls `ss` at 0.5 s intervals, exits immediately when free),
`ExecStopPost` runtime-artifact cleanup, journald-only logging
(`StandardOutput=journal`, `StandardError=journal`, `SyslogIdentifier=llama-%i`).

GPU cgroup note: user units cannot use `DeviceAllow=` for `/dev/nvidia*`
(no privilege); GPU access comes from the logind seat ACL. If converted to a
system unit, add `DeviceAllow=/dev/nvidia0 rw`, `/dev/nvidiactl rw`,
`/dev/nvidia-uvm rw` and `Nice=-5`.

## Install

```bash
mkdir -p ~/.config/systemd/user
cp /home/thomas/work/llama-serve-modern/systemd/llama@.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

## Enable / start individual models

```bash
systemctl --user enable --now llama@lfm-8b        # port 8080 (bot contract)
systemctl --user enable --now llama@lfm-1.2b      # port 8081 (bot contract)
systemctl --user enable --now llama@lfm-embedding # 8085
systemctl --user enable --now llama@lfm-colbert   # 8086
systemctl --user enable --now llama@lfm2-vl-450m  # 8088
# swap pool (enable on demand): lfm-8b-base(8090) vibethinker-3b(8082)
#   fastcontext-rl(8083) fastcontext-sft(8084) lfm2-vl-1.6b(8089)
```

## Watchdog timer (optional but recommended)

```bash
cat > ~/.config/systemd/user/llama-health.service <<'UNIT'
[Unit]
Description=llama health watchdog
[Service]
Type=oneshot
ExecStart=/home/thomas/work/llama-serve-modern/systemd/llama-health.sh
UNIT
cat > ~/.config/systemd/user/llama-health.timer <<'UNIT'
[Unit]
Description=llama health watchdog timer
[Timer]
OnBootSec=2min
OnUnitActiveSec=1min
[Install]
WantedBy=timers.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now llama-health.timer
```

## journalctl helpers

```bash
# Follow one model
journalctl --user -u llama@lfm-8b -f

# Everything llama, last hour
journalctl --user -u 'llama@*' --since '1 hour ago'

# Boot-to-now errors only
journalctl --user -u 'llama@*' -p err -b

# Watchdog decisions
journalctl --user -u llama-health.service --since today

# Confirm the GPU actually engaged (shim-defeat check — look for CUDA, not 'CPU')
journalctl --user -u llama@lfm-8b --since '-5 min' | grep -i -E 'cuda|vulkan|offload'
```

## Notes

- Env files live in the repo (`env/`) and are referenced via
  `EnvironmentFile=-/home/thomas/work/llama-serve-modern/env/%i.env`; edit and
  `systemctl --user restart llama@<name>` — no daemon-reload needed for env
  changes.
- `EnvironmentFile` path is absolute; if the repo moves, update the unit.
