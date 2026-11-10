# ptc-statecraft

A Python workspace for Statecraft's generated `hidden_dag` and `repair_chain` tasks.
Use `from ptc import task, tools, finish` in `/workspace`, then call `finish(answer)`.
Only the `python` tool is model-visible; hidden task state is not written to the sandbox.

## Evaluate

From the repository root, install the workspace and configure Prime inference
access. These evaluations consume model credits; PTC also uses a sandbox.

```sh
uv sync --locked --all-packages --group dev
uv run prime eval run ptc-statecraft \
  -a '{"config":{"taskset":{"family":"hidden_dag","max_examples":5}}}'
```

Arguments use the Verifiers v1 `config.taskset` shape. Set
`config.harness.max_turns` for an explicit per-row turn cap.

See [shared configuration and metrics](../REFERENCE.md) and the
[repository quickstart](../../README.md#try-a-task-offline) for offline oracle checks.
