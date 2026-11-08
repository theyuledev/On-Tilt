import re
import shutil
import subprocess


def _read_int(out: str) -> int | None:
    m = re.search(r"=\s*(-?\d+)", out)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", out)
    return int(nums[-1]) if nums else None


def exposure_problem(rig: dict, run=subprocess.run) -> str | None:
    # set_exposure never sees what uvc-util said, so ask the camera what it holds
    tool = shutil.which("uvc-util")
    if tool is None:
        return "exposure unverified: uvc-util not on PATH"
    got = {}
    for name in ("exposure-time-abs", "auto-exposure-mode"):
        r = run([tool, "-I", "0", "-g", name], capture_output=True, text=True)
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
