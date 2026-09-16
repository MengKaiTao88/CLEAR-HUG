#!/usr/bin/env python3
"""Deploy and launch the CPSC seed-42 single-crop diagnostic on node 10092."""
from __future__ import annotations

import argparse
import shlex
import time

from deploy_and_start import CODE_DIRNAME, connect, upload_code
from protocol import NODES

AUDIT = "q1-stmem-cpsc-seed42-single-crop-audit-20260916"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=tuple(NODES), default="10110")
    args = parser.parse_args()
    node = args.node; root = NODES[node][3]; client = connect(node)
    try:
        upload_code(client, root)
        result = f"{root}/results/{AUDIT}"
        script = f"{root}/src/CLEAR-HUG/experiments/{CODE_DIRNAME}/run_single_crop_audit.py"
        python = f"{root}/.venv/bin/python"
        # Anchor the argv so the remote wrapper shell's own command line cannot
        # satisfy the guard merely because it contains the launch command.
        pattern = f"^{python} {script} --root {root}$"
        command = (f"mkdir -p {shlex.quote(result)}; "
                   f"if ! pgrep -af {shlex.quote(pattern)} >/dev/null; then "
                   f"nohup setsid " + ("env LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 " if node == "10110" else "") +
                   f"{python} {script} --root {root} > {result}/run.log 2>&1 < /dev/null & "
                   f"echo $! > {result}/run.pid; fi")
        _, stdout, _ = client.exec_command(command); time.sleep(1); stdout.channel.close()
        print(f"started {node}:{result}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
