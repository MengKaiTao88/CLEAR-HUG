# HeartLang / ST-MEM frozen-encoder benchmark

This campaign evaluates the pinned official HeartLang and ST-MEM encoders with
one shared multilabel linear-probe implementation. Encoder features are fixed;
only `Linear(768, classes)` is trained. Train/validation caches are created
before selection. Test data is inaccessible until the global pre-test gate
passes.

The frozen protocol is defined in `protocol.py`. Server artifacts live below
`/root/107552503710*` in accordance with the workspace policy.
