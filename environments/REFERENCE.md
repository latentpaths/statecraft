# Statecraft environment configuration

Both adapters generate `hidden_dag` and `repair_chain` tasks locally.

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
| `calls` | Number of Statecraft tool calls |
| `budget` | Tool-call budget |
| `reference_calls` | Public-solver call count |
| `progress` | Completion-aligned diagnostic progress |
| `failed_calls` | Budget-consuming calls with a failure code |
