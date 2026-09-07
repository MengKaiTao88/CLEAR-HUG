# ST-MEM and HeartLang deployment

This directory records the reproducible deployment of the official ST-MEM and
HeartLang repositories on the 10110 server.  All server-side artifacts live
under `/root/107552503710`.

## Pinned upstream revisions

- ST-MEM: `311c6446894dee4126db8d2520024e9f62cf1616`
- HeartLang: `a08afd8117e813fee212a1ac283db4a7140a4669`

## Official weights

- ST-MEM encoder: `st_mem_vit_base_encoder.pth`
- HeartLang pretraining checkpoint: `checkpoint-200.pth`
- HeartLang VQ-HBR tokenizer: `vqhbr-checkpoint-100.pth`

The generated `deployment-manifest.json` is the authoritative record of source
and weight hashes, runtime versions, compatibility changes, and smoke-test
results.  Run `smoke_test.py` inside the server workspace after the official
repositories and weights have been installed.

ST-MEM is distributed under CC BY-NC 4.0 and is restricted to non-commercial
use. HeartLang includes an MIT license in the upstream `LICENCE` file.

The deployed runtime command must preload the NVIDIA driver shim on this host:

```bash
env LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 \
  /root/107552503710/.venv/bin/python <command>
```
