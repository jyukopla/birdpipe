import subprocess
import time
import re
from typing import Optional, Dict


def _run(cmd: list[str], timeout: float = 3.0) -> str:
    """Run a system command and return its stdout as text."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, timeout=timeout
    )
    return result.stdout


def _parse_tracking(text: str) -> Dict[str, Optional[str]]:
    """Extract key metrics (both raw strings and floats) from 'chronyc tracking' output."""
    metrics: Dict[str, Optional[str]] = {
        "system_time_raw": None,
        "system_offset_s": None,
        "last_offset_s": None,
        "rms_offset_s": None,
    }

    for line in text.splitlines():
        if line.startswith("System time"):
            metrics["system_time_raw"] = line.split(":", 1)[1].strip()
            m = re.search(r"([-0-9.eE]+)\s*seconds", line)
            if m:
                metrics["system_offset_s"] = float(m.group(1))
        elif line.startswith("Last offset"):
            m = re.search(r"([-0-9.eE]+)\s*seconds", line)
            if m:
                metrics["last_offset_s"] = float(m.group(1))
        elif line.startswith("RMS offset"):
            m = re.search(r"([-0-9.eE]+)\s*seconds", line)
            if m:
                metrics["rms_offset_s"] = float(m.group(1))

    return metrics


def _format_report(metrics: Dict[str, Optional[str]]) -> str:
    """Return a short human-readable accuracy summary with quality rating."""
    def fmt_us(val: Optional[float]) -> str:
        return f"{val*1e6:.1f} µs ({val*1e9:.0f} ns)" if val is not None else "N/A"

    sys_raw = metrics.get("system_time_raw") or "N/A"
    sys_off = fmt_us(metrics.get("system_offset_s"))
    last = fmt_us(metrics.get("last_offset_s"))
    rms = fmt_us(metrics.get("rms_offset_s"))

    # Quality classification based on RMS offset
    quality = "Unknown"
    try:
        rms_val = metrics.get("rms_offset_s")
        if rms_val is not None:
            if rms_val <= 5e-6:
                quality = "Excellent"
            elif rms_val <= 50e-6:
                quality = "Good"
            elif rms_val <= 500e-6:
                quality = "Fair"
            else:
                quality = "Poor"
    except Exception:
        pass

    return (
        "Clock accuracy after PPS lock:\n"
        f"  • System time (raw): {sys_raw}\n"
        f"  • Current offset:    {sys_off}\n"
        f"  • Last offset:       {last}\n"
        f"  • RMS offset:        {rms}\n"
        f"  • Quality rating:    {quality}"
    )


def get_pps_status() -> bool:
    """
    Check if PPS source is present in chrony.
    Returns True if PPS is detected, False otherwise.
    Prints a short accuracy summary when PPS is active.
    """
    for attempt in range(3):
        try:
            out = _run(["chronyc", "sources"], timeout=3)
            pps_ok = any("PPS" in line and ("*" in line or "+" in line) for line in out.splitlines())

            if pps_ok:
                print("PPS signal detected")

                try:
                    tracking = _run(["chronyc", "tracking"], timeout=3)
                    metrics = _parse_tracking(tracking)
                    print(_format_report(metrics))
                except Exception as e:
                    print(f"Could not read chrony tracking: {e}")

                return True
        except Exception as e:
            print(f"PPS check failed (attempt {attempt+1}): {e}")

        print(f"No PPS yet (attempt {attempt+1}/3). Retrying...")
        time.sleep(2)

    return False
