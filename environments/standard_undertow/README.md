# standard-undertow

### Overview
- **Environment ID**: `standard-undertow`
- **Short description**: Undertow benchmark tasks exposed through native tool calls.
- **Tags**: undertow, tool-use, eval

### Datasets
- **Primary dataset(s)**: Generated Undertow task families (`hidden_dag`, `repair_chain`).
- **Source links**: Generated locally by the `undertow` package.
- **Split sizes**: Controlled by `max_examples`.

### Task
- **Type**: multi-turn tool use
- **Output format expectations**: Call native task tools to inspect hidden state, then call `submit` with the final answer.
- **Rubric overview**: Uses shaped `reward`, exact `solved`, and compact trace metrics.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run standard-undertow
```

Configure model and sampling:

```bash
prime eval run standard-undertow \
  -a '{"config":{"taskset":{"family":"hidden_dag","max_examples":5}}}' \
  -m openai/gpt-4.1-mini -n 20 -r 1 -t 1024 -T 0.7
```

Notes:
- CLI `-a` values use Verifiers v1 config shape under `config.taskset`.
- `harness.max_turns` overrides the default per-task `budget + 4`.

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
Each task row defaults to `max_turns = task.budget + 4`. Set
`config.harness.max_turns` to override that value for generated rows.

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Shaped benchmark score |
| `solved` | Whether the submitted answer matched the task target |
| `calls` | Number of Undertow tool calls |
| `budget` | Tool-call budget |
| `reference_calls` | Public-solver call count |
| `efficiency` | Call efficiency relative to `reference_calls` |
| `progress` | Family-specific partial progress |
| `state_changes` | Calls that changed environment state |
| `failure_calls` | Calls with a failure code |
