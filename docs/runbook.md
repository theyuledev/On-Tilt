# rig runbook

one page for the person at the rig. every command below is typed from the repo root with

```
export UV_PROJECT_ENVIRONMENT=".venv.nosync"
```

(or the absolute path to the shared env). the app and the bench both open the serial port, so only one of them runs at a time.

## start

configs/settings.local.toml (gitignored) holds the rig-only bits:

```toml
[rig]
actuator = "real"
serial_port = "/dev/cu.usbmodem..."
```

- app: `uv run python -m src.app`, page at http://127.0.0.1:8000 (PORT env or .env changes it). it prints `hand input off: ...` / `camera off: ...` when those are missing and keeps running without them
- bench: `uv run python -m tools.bench [port]`. the optional port overrides serial_port for this run

## arm / kill

page: `arm` button arms, `kill (space)` button or the space key kills. arrows / wasd steer the plate while armed. the pill says disarmed / armed / killing, `board lost` shows up when the port is gone.

bench, typed at the `>` prompt:

- `l` loads the settings limits and level pulse (`L xlo xhi ylo yhi xlevel ylevel`), also after an `l <n>` widened them, then logs the headroom to the board envelope (1050..1950 us)
- `l 200` same but with limits of +-200 us around center on both axes. bench only, settings are not written, and the bench keeps them only when the board answers `OK L`; the next bare `l` is back on the settings limits
- `A` arms, pulses start at the level pulse (center + trims) and the bench heartbeat starts there, so arrows work right away
- `P x y` sets pulses, the reply shows the clamped values; the bench keeps resending the last P as a heartbeat
- arrow keys nudge the live pulses 5 us: left/right = x -/+, up/down = y +/-
- status line: `pulses (x, y) off (+dx, +dy) us` is the offset from center
- space = kill: ramps to level at kill_ramp_us_s, then `D`. a second space mid ramp does nothing
- ctrl-c = same ramp, then exit
- `D` disarms right away, pulses stop and the servos hold wherever they were
- `quiet` stops the heartbeat so you can watch the board level itself after its 250 ms timeout
- `S` status: `OK S <armed> <limits> <ms since last cmd> <fw>`

## reset drill

1. kill (space), wait for `no pulses` in the status line
2. servo power off
3. unplug and replug the board usb
4. board boots disarmed: in the bench `S` should answer `OK S 0 0 ...`
5. servo power on
6. restart the app or the bench

servo power is always off for any firmware upload.

## camera revive

the overhead cam jams after repeated open/close. the app says `camera off: ...` at start when it could not open it and runs on without video.

```
uvc-util -I 0 -s auto-exposure-mode=8
uvc-util -I 0 -s auto-exposure-mode=1 -s exposure-time-abs=30
```

(30 = 3 ms, units are 100 us), then restart the app. still dead: replug the camera and try again. never set width/height on this cam, it stops sending frames.

## board lost

the page shows `board lost` when the serial port vanished or answered garbage. the loop stops sending and will not re-arm until the app is restarted. the board stops hearing us, ramps to level within ~250 ms + ramp and stops pulses on its own. do the reset drill.

`board disarmed: ...` in the app log is the softer case: the board answered ERR, or a P got no reply and `S` said the board is not armed any more (a missed reply with the board still armed is carried on, nothing disarms). the loop disarms, the page can arm again.

## limits + trims slot

servos powered, bench running, app not running. cx, cy = 1500 + current trims; the `off` readout already subtracts them.

1. `l`, `A`. plate comes up near level at center
2. trims: phone level on the plate. arrow keys nudge 5 us until both axes read 0.0. the status line `off (+dx, +dy)` is trim_x_us / trim_y_us
3. us per plate degree, x: from the trimmed level send `P <x+100> <y>`, read the plate tilt t1 on the phone, then `P <x-100> <y>`, read t2. us_per_deg_x = 200 / (|t1| + |t2|). write down which way +100 tilts the plate (sign). same on y with the second number
4. limits: `D`, `l 200`, `A`, then `P` to the four corners (center +-200 on both axes). the bench sends P with no slew, so a corner is a 200+ us jump: approach each new corner in two or three P steps (e.g. +100, then +150, then the corner) and watch between steps, only the final four-corner pass goes corner to corner. watch for anything binding (linkage, u-joint, plate hitting the frame) and listen for a servo buzzing at its end stop. `D`, `l 250`, `A`, corners again, and so on in 50 us steps. stop at the last range where nothing touched, that is limit_x_us / limit_y_us (per axis if they differ). four corner check at the final numbers. the headroom line says how far the envelope still allows
5. `D`, ctrl-c

the numbers go into configs/settings.toml [rig] in their own PR: trim_x_us, trim_y_us, us_per_deg_x, us_per_deg_y, limit_x_us, limit_y_us (startup_range_us / startup_flag once the limits are trusted), each labelled `measured (2026-11-07)`. nothing is typed into settings before it was read off the rig.
