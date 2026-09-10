# Repository content policy

Only source code, development/build scripts, small solver configuration files,
and human-written documentation belong in Git. Do not commit generated experiment
data, result/check JSON, receipts, input inventories, CSV, logs, dumps, geometry
meshes, plots, archives, executables, or compiled-source snapshots. Compressing
an artifact does not make it source code.

Store experiment artifacts under `D:/CirrusExperiments/cirrus-amg` and transfer
them separately by local media. Existing ignored files in the worktree are local
data; preserve them unless the user asks for removal. Do not use `git add -f` to
bypass the content policy or add an artifact through a misleading file suffix.

Before committing, run:

```text
python scripts/check_git_content.py --staged
```

Before preparing this branch for transfer, also run:

```text
python scripts/check_git_content.py --base origin/main
```

The shared main branch and existing tags retain their original history. Keep new
development commits free of data. If an unpushed development history needs
cleanup, first preserve and verify a local backup; do not rewrite shared refs.

The twisted-tube solver goal is currently paused at the user's request. Do not
start or resume large flow experiments while doing repository/documentation work.
See `validation/twisted/REPRODUCE_ON_LARGE_MACHINE.md` for the pending accuracy
checks and the separate local input package.
