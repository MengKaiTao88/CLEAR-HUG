#!/usr/bin/env python3
"""Deploy and launch the CPSC seed-42 single-crop diagnostic on node 10092."""
from __future__ import annotations

import shlex
import time

from audit_stmem_cpsc_seed42 import AUDIT
from deploy_and_start import CODE_DIRNAME, connect, upload_code
from protocol import NODES


def main() -> None:
    node = "10092"; root = NODES[node][3]; client = connect(node)
    try:
        upload_code(client, root)
        result = f"{root}/results/{AUDIT}"
        script = f"{root}/src/CLEAR-HUG/experiments/{CODE_DIRNAME}/run_single_crop_audit.py"
        python = f"{root}/.venv/bin/python"
        pattern = f"[r]un_single_crop_audit.py --root {root}"
        command = (f"mkdir -p {shlex.quote(result)}; "
                   f"if ! pgrep -af {shlex.quote(pattern)} >/dev/null; then "
                   f"nohup setsid {python} {script} --root {root} > {result}/run.log 2>&1 < /dev/null & "
                   f"echo $! > {result}/run.pid; fi")
        _, stdout, _ = client.exec_command(command); time.sleep(1); stdout.channel.close()
        print(f"started {node}:{result}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
