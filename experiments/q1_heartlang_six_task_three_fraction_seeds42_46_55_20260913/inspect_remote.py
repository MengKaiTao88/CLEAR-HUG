#!/usr/bin/env python3
"""Read-only audit of the three HeartLang target nodes."""
from __future__ import annotations

import concurrent.futures
import argparse
import json
import sys
from pathlib import Path

import paramiko

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


WORKSPACE = Path(__file__).resolve().parents[4]
NODES = (
    ("10110", ".env.remote", "10.109.118.172", 10110, "/root/107552503710"),
    ("10092", ".env.remote.secondary", "10.109.118.204", 10092, "/root/107552503710-1"),
    ("10103", ".env.remote.tertiary", "10.109.118.204", 10103, "/root/107552503710-2"),
)


def env_file(name: str) -> dict[str, str]:
    values = {}
    for raw in (WORKSPACE / name).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def inspect(node: tuple[str, str, str, int, str]) -> dict:
    name, env_name, host, port, root = node
    cfg = env_file(env_name)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            host,
            port=port,
            username=cfg.get("CODEX_REMOTE_USER", "root"),
            password=cfg["CODEX_REMOTE_PASSWORD"],
            timeout=20,
            banner_timeout=20,
            auth_timeout=20,
        )
    except Exception as exc:
        client.close()
        return {"node": name, "connect_error": f"{type(exc).__name__}: {exc}"}
    command = rf'''set -o pipefail
echo NODE={name}
echo ROOT={root}
test -d {root}/external_models/HeartLang && echo HEARTLANG_DIR=yes || echo HEARTLANG_DIR=no
if test -d {root}/external_models/HeartLang/.git; then git -C {root}/external_models/HeartLang rev-parse HEAD; fi
test -d {root}/external_models/ST-MEM && echo STMEM_DIR=yes || echo STMEM_DIR=no
if test -d {root}/external_models/ST-MEM/.git; then git -C {root}/external_models/ST-MEM rev-parse HEAD; fi
for f in {root}/model_weights/HeartLang/checkpoint-200.pth {root}/model_weights/HeartLang/vqhbr-checkpoint-100.pth; do
  if test -f "$f"; then sha256sum "$f"; else echo MISSING=$f; fi
done
if test -f {root}/model_weights/ST-MEM/st_mem_vit_base_encoder.pth; then
  sha256sum {root}/model_weights/ST-MEM/st_mem_vit_base_encoder.pth
else
  echo MISSING={root}/model_weights/ST-MEM/st_mem_vit_base_encoder.pth
fi
test -d {root}/campaign-inputs/q1-clear-deepsets-hilar-10seed-20260903/ecg_datasets-resolved-v2 && echo DATA=yes || echo DATA=no
for py in {root}/.venv/bin/python /root/miniconda3/envs/py3.10torch2.1.0cu121/bin/python /root/miniconda3/envs/py3.10torch2.1.2cu121/bin/python; do test -x "$py" && echo PYTHON=$py; done
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
df -BG --output=avail {root} | tail -1
ps -eo pid,args | grep -E '[p]ython.*(q1_|HeartLang|run_class)' | tail -10
for rel in PTBXL/superdiagnostic/data PTBXL/subdiagnostic/data PTBXL/form/data PTBXL/rhythm/data CPSC2018/data CSN/data; do
  p={root}/src/CLEAR-HUG/datasets/ecg_datasets/$rel
  if test -d "$p"; then
    echo RAW=$rel
    find "$p" -maxdepth 1 -type f -printf '%f:%s\\n' | sort
  else
    echo RAW_MISSING=$rel
  fi
done
if test "{name}" = 10110; then
  find {root}/external_models/HeartLang -maxdepth 3 -type f \
    \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.md' \) \
    | sort | head -200
  find {root}/external_models/ST-MEM -maxdepth 4 -type f \
    \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.md' \) \
    | sort | head -300
fi
'''
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    client.close()
    return {"node": name, "stdout": out, "stderr": err}


def cat_source(relative_path: str, line_range: str | None = None, repository: str = "HeartLang") -> None:
    name, env_name, host, port, root = NODES[0]
    cfg = env_file(env_name)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=cfg.get("CODEX_REMOTE_USER", "root"),
                   password=cfg["CODEX_REMOTE_PASSWORD"], timeout=20)
    sftp = client.open_sftp()
    try:
        path = f"{root}/external_models/{repository}/{relative_path}"
        with sftp.open(path, "r") as handle:
            body = handle.read().decode(errors="replace")
        if line_range:
            start, end = (int(value) for value in line_range.split(":", 1))
            body = "\n".join(body.splitlines()[start - 1:end])
        print(body)
    finally:
        sftp.close()
        client.close()


def checkpoint_keys() -> None:
    name, env_name, host, port, root = NODES[0]
    cfg = env_file(env_name)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=cfg.get("CODEX_REMOTE_USER", "root"),
                   password=cfg["CODEX_REMOTE_PASSWORD"], timeout=20)
    python = f"{root}/.venv/bin/python"
    code = (
        "import torch; "
        f"p=torch.load('{root}/model_weights/HeartLang/checkpoint-200.pth',map_location='cpu'); "
        "print(type(p), list(p) if isinstance(p,dict) else 'not-dict'); "
        "s=p.get('model',p.get('state_dict',p)) if isinstance(p,dict) else p; "
        "print(len(s)); print('\\n'.join(list(s)[:30])); print('LAST'); print('\\n'.join(list(s)[-10:]))"
    )
    _, stdout, stderr = client.exec_command(f'{python} -c "{code}"')
    print(stdout.read().decode(errors="replace"))
    error = stderr.read().decode(errors="replace")
    if error:
        print(error, file=sys.stderr)
    client.close()


def qrs_shapes() -> None:
    name, env_name, host, port, root = NODES[0]
    cfg = env_file(env_name)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=cfg.get("CODEX_REMOTE_USER", "root"),
                   password=cfg["CODEX_REMOTE_PASSWORD"], timeout=20)
    python = f"{root}/.venv/bin/python"
    base = f"{root}/campaign-inputs/q1-clear-deepsets-hilar-10seed-20260903/ecg_datasets-resolved-v2"
    code = (
        "import numpy as np,pathlib; b=pathlib.Path('" + base + "'); "
        "rels=['PTBXL_QRS/superdiagnostic','PTBXL_QRS/subdiagnostic','PTBXL_QRS/form','PTBXL_QRS/rhythm','CPSC2018_QRS/data','CSN_QRS/data']; "
        "[(print(r, np.load(b/r/'train_data.npy',mmap_mode='r').shape, np.load(b/r/'train_data_in_chans.npy',mmap_mode='r').shape)) for r in rels]"
    )
    _, stdout, stderr = client.exec_command(f'{python} -c "{code}"')
    print(stdout.read().decode(errors="replace"))
    error = stderr.read().decode(errors="replace")
    if error:
        print(error, file=sys.stderr)
    client.close()


def split_leakage_audit() -> None:
    """Check record identifiers and exact waveform hashes across every split."""
    name, env_name, host, port, root = NODES[0]
    cfg = env_file(env_name)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=cfg.get("CODEX_REMOTE_USER", "root"),
                   password=cfg["CODEX_REMOTE_PASSWORD"], timeout=20)
    python = f"{root}/.venv/bin/python"
    code = rf'''
import hashlib, json, pathlib
import numpy as np

base = pathlib.Path("{root}/src/CLEAR-HUG/datasets/ecg_datasets")
tasks = {{
    "superdiagnostic": "PTBXL/superdiagnostic/data",
    "subdiagnostic": "PTBXL/subdiagnostic/data",
    "form": "PTBXL/form/data",
    "rhythm": "PTBXL/rhythm/data",
    "cpsc2018": "CPSC2018/data",
    "csn": "CSN/data",
}}
report = {{}}
for task, relative in tasks.items():
    directory = base / relative
    ids = {{}}
    hashes = {{}}
    waveform_labels = {{}}
    for split in ("train", "val", "test"):
        values = np.load(directory / f"{{split}}_path.npy", allow_pickle=True)
        ids[split] = set(map(str, values.tolist()))
        waveforms = np.load(directory / f"{{split}}_data.npy", mmap_mode="r")
        labels = np.load(directory / f"{{split}}_labels.npy", mmap_mode="r")
        mapping = {{}}
        for row, label in zip(waveforms, labels):
            digest = hashlib.sha256(np.ascontiguousarray(row).view(np.uint8)).digest()
            mapping.setdefault(digest, set()).add(np.ascontiguousarray(label).tobytes())
        waveform_labels[split] = mapping
        hashes[split] = set(mapping)
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    label_agreement = {{}}
    for left, right in pairs:
        shared = hashes[left] & hashes[right]
        same = sum(bool(waveform_labels[left][key] & waveform_labels[right][key]) for key in shared)
        label_agreement[f"{{left}}_{{right}}"] = {{
            "shared_waveforms": len(shared),
            "same_label": same,
            "conflicting_only": len(shared) - same,
        }}
    report[task] = {{
        "id_unique": {{split: len(ids[split]) for split in ids}},
        "id_overlap": {{
            "train_val": len(ids["train"] & ids["val"]),
            "train_test": len(ids["train"] & ids["test"]),
            "val_test": len(ids["val"] & ids["test"]),
        }},
        "exact_waveform_overlap": {{
            "train_val": len(hashes["train"] & hashes["val"]),
            "train_test": len(hashes["train"] & hashes["test"]),
            "val_test": len(hashes["val"] & hashes["test"]),
        }},
        "cross_split_waveform_label_agreement": label_agreement,
        "within_split_duplicate_waveforms": {{
            split: len(np.load(directory / f"{{split}}_data.npy", mmap_mode="r")) - len(hashes[split])
            for split in hashes
        }},
    }}
print(json.dumps(report, sort_keys=True))
'''
    command = f"{python} - <<'PY'\n{code}\nPY"
    _, stdout, stderr = client.exec_command(command)
    print(stdout.read().decode(errors="replace"))
    error = stderr.read().decode(errors="replace")
    if error:
        print(error, file=sys.stderr)
    client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", help="print one HeartLang source file from 10110")
    parser.add_argument("--stmem-cat", help="print one ST-MEM source file from 10110")
    parser.add_argument("--lines", help="one-based inclusive line range, e.g. 300:500")
    parser.add_argument("--checkpoint-keys", action="store_true")
    parser.add_argument("--qrs-shapes", action="store_true")
    parser.add_argument("--split-leakage-audit", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_keys:
        checkpoint_keys()
        return
    if args.qrs_shapes:
        qrs_shapes()
        return
    if args.split_leakage_audit:
        split_leakage_audit()
        return
    if args.cat:
        cat_source(args.cat, args.lines)
        return
    if args.stmem_cat:
        cat_source(args.stmem_cat, args.lines, repository="ST-MEM")
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        reports = list(pool.map(inspect, NODES))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
