#!/usr/bin/env python3
"""Deploy and launch the isolated MERL-style ST-MEM CPSC seed-42 run."""
from __future__ import annotations

import shlex
import time

from deploy_and_start import CODE_DIRNAME, NODES, connect, upload_code
CAMPAIGN = "q1-stmem-cpsc-merl-protocol-seed42-20260916"


def main() -> None:
    node = "10110"; root = NODES[node][3]; client = connect(node)
    try:
        upload_code(client, root)
        result = f"{root}/results/{CAMPAIGN}"
        python = f"{root}/.venv/bin/python"
        script = f"{root}/src/CLEAR-HUG/experiments/{CODE_DIRNAME}/run_stmem_merl_cpsc_seed42.py"
        pattern = f"^{python} {script} --root {root}$"
        command = (
            f"mkdir -p {shlex.quote(result)}; "
            f"if ! pgrep -f {shlex.quote(pattern)} >/dev/null; then "
            f"nohup setsid env LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 "
            f"{python} {script} --root {root} > {result}/run.log 2>&1 < /dev/null & "
            f"echo $! > {result}/run.pid; fi"
        )
        _, stdout, _ = client.exec_command(command); time.sleep(1); stdout.channel.close()
        print(f"started {node}:{result}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
