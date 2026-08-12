"""Master verification suite runner for QuantEdge AI Gateway."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

VERIFICATION_SCRIPTS = [
    "verify_structure.py",
    "verify_indicators.py",
    "verify_quality.py",
    "verify_persistence.py",
    "verify_regime_mtf.py",
    "verify_scanner.py",
    "verify_llm.py",
    "verify_signal_settlement.py",
    "verify_mcp.py",
    "verify_api.py",
    "verify_memory_bot.py",
    "verify_review_reconcile.py",
]


def main() -> int:
    print("=" * 80)
    print("QUANTEDGE AI -- MASTER INTEGRATION VERIFICATION PASS")
    print("=" * 80)

    python_exe = sys.executable
    scripts_dir = Path(__file__).resolve().parent
    failed_scripts: list[str] = []

    for script_name in VERIFICATION_SCRIPTS:
        script_path = scripts_dir / script_name
        print(f"\n>>> Running {script_name}...")
        # check=False on purpose: a failing suite must not abort the run. The whole
        # point is to report every suite's result, so the exit code is collected
        # below rather than raised here.
        proc = subprocess.run(  # noqa: S603  # fixed argv, our own interpreter and scripts
            [python_exe, str(script_path)], capture_output=False, check=False
        )
        if proc.returncode != 0:
            failed_scripts.append(script_name)

    print("\n" + "=" * 80)
    if failed_scripts:
        print(f"FAILED: {len(failed_scripts)} verification script(s) failed:")
        for s in failed_scripts:
            print(f"  - {s}")
        return 1

    print("SUCCESS: ALL VERIFICATION SUITES PASSED CLEANLY (12/12)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
