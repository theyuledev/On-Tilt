import re
import shutil
import subprocess


def _read_int(out: str) -> int | None:
    m = re.search(r"=\s*(-?\d+)", out)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", out)
    return int(nums[-1]) if nums else None


def uvc_target(rig: dict, tool: str, run=subprocess.run) -> list[str] | str:
    """uvc-util args that select the overhead cam, or why there are none."""
    if rig.get("camera_uvc"):
        return ["-V", rig["camera_uvc"]]
    out = run([tool, "-d"], capture_output=True, text=True).stdout or ""
    rows = [w[0] for w in map(str.split, out.splitlines()) if w and w[0].isdigit()]
    if len(rows) == 1:
        return ["-I", rows[0]]
    if rows:
        return "several uvc cameras plugged in, set camera_uvc in configs/settings.local.toml"
    return "no uvc camera listed"


def uvc_command(rig: dict, run=subprocess.run) -> list[str] | None:
    tool = shutil.which("uvc-util")
    if tool is None:
        return None
    target = uvc_target(rig, tool, run)
    if isinstance(target, str):
        print(f"camera: {target}")
        return None
    return [tool, *target]


def exposure_problem(rig: dict, run=subprocess.run) -> str | None:
    # set_exposure never sees what uvc-util said, so ask the camera what it holds
    tool = shutil.which("uvc-util")
    if tool is None:
        return "exposure unverified: uvc-util not on PATH"
    target = uvc_target(rig, tool, run)
    if isinstance(target, str):
        return f"exposure unverified: {target}"
    got = {}
    for name in ("exposure-time-abs", "auto-exposure-mode"):
        r = run([tool, *target, "-g", name], capture_output=True, text=True)
        value = None
        if r.returncode == 0:
            value = _read_int(r.stdout or "")
            if value is None:
                value = _read_int(r.stderr or "")
        if value is None:
            tail = ((r.stdout or "").strip() or (r.stderr or "").strip())[-60:]
            return f"exposure unverified: {name}: {tail}"
        got[name] = value
    want = round(rig["camera_exposure_ms"] * 10)
    problems = []
    if got["exposure-time-abs"] != want:
        problems.append(f"exposure {got['exposure-time-abs']} (100 us steps), wanted {want}")
    if got["auto-exposure-mode"] != 1:
        problems.append(f"auto exposure mode {got['auto-exposure-mode']}, wanted 1")
    return "; ".join(problems) or None
