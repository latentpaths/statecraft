# ptc-undertow

### Overview
- **Environment ID**: `ptc-undertow`
- **Short description**: Undertow benchmark tasks exposed through a Verifiers v1 PTC Python workspace.
- **Tags**: undertow, programmatic-tool-calling, eval

### Datasets
- **Primary dataset(s)**: Generated Undertow task families (`hidden_dag`, `repair_chain`).
- **Source links**: Generated locally by the `undertow` package.
- **Split sizes**: Controlled by `max_examples`.

### Task
- **Type**: multi-turn programmatic tool use
- **Output format expectations**: Use the single Verifiers `python` tool to build durable files in `/workspace`, import `from ptc import task, tools, finish`, call primitive Undertow tools through the private host RPC bridge, then call `finish(answer)`.
- **Rubric overview**: Uses solve-gated `reward`, exact `solved`, and compact trace metrics.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run ptc-undertow
```

Configure model and sampling:

```bash
prime eval run ptc-undertow \
  -a '{"config":{"taskset":{"family":"hidden_dag","max_examples":5}}}' \
  -m openai/gpt-4.1-mini -n 20 -r 1 -t 1024 -T 0.7
```

Notes:
- CLI `-a` values use Verifiers v1 config shape under `config.taskset`.
- Only `python(code: str)` is model-visible. The sandbox `ptc` package uses a private endpoint route for Undertow primitives.
- Hidden task state is never written to the sandbox.

### Taskset Config

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `family` | string | unset | Single task family, such as `hidden_dag` or `repair_chain` |
| `families` | list[string] | unset | Families to cycle through when generating tasks |
| `seed` | int | `0` | First task seed |
| `max_examples` | int | `1` | Number of generated examples |
| `shuffle` | bool | `false` | Shuffle generated tasks |
| `task_params` | table | unset | Family-specific generation parameters |

### Harness Config
Task rows do not set `max_turns` by default. Set `config.harness.max_turns`
to add an explicit per-row turn cap.

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Primary solve-gated score |
| `solved` | Whether the submitted answer matched the task target |
| `calls` | Number of Undertow tool calls |
| `budget` | Tool-call budget |
| `reference_calls` | Public-solver call count |
| `progress` | Completion-aligned diagnostic progress |
| `failed_calls` | Budget-consuming calls with a failure code |
