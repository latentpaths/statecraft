# Undertow: A Benchmark for Programmatic Tool Calling in Hidden Stateful Environments

There is a narrow but important question hiding inside the larger conversation
about agents.

Not: can a model call tools?

Not: can a model write code?

The more interesting question is whether a model, given a programmable workspace,
can build the missing control layer around tools: memory, state tracking,
planning, verification, retry logic, and enough local abstraction to keep a
partially observed environment straight.

That is the purpose of Undertow.

Undertow is a synthetic, controllable benchmark for comparing ordinary
tool-calling agents with programmatic tool-calling agents. It asks models to
interact with hidden environments through primitive public tools. The environment
has state. Observations are partial. Actions can go stale. Some tools are
irrelevant. The target condition is hidden behind a reward checker. The model
does not get to inspect the answer, and local checks are useful only as
discipline, not as reward authority.

The bet behind the benchmark is simple: programmatic tool calling should not
matter much when a task is a short list of obvious calls. It should matter when
the task requires durable cognition.

## Why Another Tool-Use Benchmark?

Most tool-use evaluations measure the model's ability to select and call APIs.
That is a legitimate capability, but it is not the whole agent problem. Real
agent work often fails after the first correct tool call. The agent must remember
what it learned, avoid reusing stale facts, distinguish useful actions from
distractors, preserve intermediate state, and recover after an attempted action
does not do what it expected.

FuncBenchGen gives a clean example of how to study one part of this problem. It
constructs synthetic, contamination-free function-calling tasks from hidden
dependency DAGs. The model sees a set of functions, input variables, and a target
variable. It must infer the call sequence needed to compute the target. The
important thing about FuncBenchGen is not only the DAG. It is the experimental
discipline: generated tasks, controlled difficulty, hidden ground truth,
machine-checked traces, and a failure taxonomy.

Undertow follows that spirit, but changes the center of gravity.

FuncBenchGen is mostly about hidden dataflow. Undertow is about hidden state. It
keeps the contamination-free and controllable construction, then adds the thing
that makes programmatic tool calling worth testing: a workspace in which the
agent can create its own operating system for the task.

The standard agent sees native tools and must solve through the normal
conversation/tool-call loop.

The PTC agent sees Python. It can write files, build tables, cache observations,
construct a planner, run small local checks, and call the same primitive
environment tools through a small `ptc` package. The reward is still protected.
The hidden state is still protected. But the cognition no longer has to live
entirely in the model's transient context.

That is the distinction Undertow is built to isolate.

## Benchmark Scope

Undertow v0 deliberately stays small. It has two task families:

1. `hidden_dag`: deterministic hidden function-dependency tasks.
2. `repair_chain`: mutable, partially observable repair tasks.

It has two interaction modes:

1. `standard`: the model receives the native tools directly.
2. `ptc`: the model receives a Python workspace and calls tools through `ptc`.

It does not try to be a broad suite of web, browser, database, shell, and
software-engineering tasks. That would make the benchmark harder to interpret.
Undertow's role is more restrained: hold the environment fixed, control the
difficulty knobs, and ask whether a programmable workspace changes reliability.

The initial experiment compares standard and PTC agents across both task
families with a fixed generated grid:

- `hidden_dag`
  - `n_core in {5, 10}`
  - `depth in {2, 5}`
  - `n_connected_distractors in {0, 5}`
- `repair_chain`
  - `n_objects in {1, 3}`
  - `n_repair_steps in {2, 3}`
  - `n_distractor_tools in {0, 2}`
- seeds: `0..4`
- modes: `standard`, `ptc`

For both families, the default budget is three times the public-solver call
count. In `hidden_dag`, that means all generated functions plus submit. In
`repair_chain`, that means inspecting every tool once, scanning every object
once, moving once per object, applying every required repair stage, testing every
object, and submitting. The extra budget is intentional. A real model will do
discovery, repeat a scan, make an occasional wrong hypothesis, or test too early.
The benchmark should measure those choices, not collapse immediately because the
budget only fits an oracle-perfect trace.

This is intentionally not an exhaustive sweep. It is the first clean cut:
small-enough to inspect, large-enough to expose failure modes, and structured
enough to support a real comparison.

## Formalization

An Undertow task is a tuple:

```text
tau = (S0, Omega, A, T, R, B)
```

where:

- `S0` is the hidden initial state.
- `Omega` is the set of observation functions.
- `A` is the set of state-changing actions.
- `T` is the target condition.
- `R` is the reward checker.
- `B` is the tool-call budget.

The agent never observes `S0` directly. It receives a public task specification:
tool schemas, public identifiers, controllable task parameters, and the target
description. At step `t`, the agent chooses an operation:

```text
a_t in Omega union A
```

The environment applies the transition:

```text
S_{t+1} = transition(S_t, a_t)
```

and returns a public observation:

```text
o_t = observe(S_{t+1}, a_t)
```

The episode ends when the agent submits an answer or exhausts the budget. Success
is defined only by the reward checker:

```text
R(S_T, answer) = 1
```

This matters. A model can maintain a local verifier. It can write a plan. It can
declare that it is done. None of that is authoritative. The environment decides.

Every call is recorded as a trace event:

```json
{
  "order": 7,
  "tool_name": "apply_tool",
  "args": {
    "tool_id": "tool_3",
    "object_id": "machine_0",
    "observation_version": 1
  },
  "observation": {
    "ok": true,
    "object_id": "machine_0",
    "repair_stage": 2,
    "status": "broken",
    "observation_version": 2
  },
  "failure": null,
  "state_delta": {
    "machine_0": {
      "repair_stage": {"from": 1, "to": 2},
      "status": "broken"
    }
  },
  "consumed_budget": true
}
```

The trace is the basis for scoring and analysis. It lets the benchmark classify
failures without human annotation: incorrect value, unknown reference, stale
observation, violated constraint, distractor action, repeated call, premature
submit, and budget exhaustion.

## Interaction Modes

Undertow's comparison is only meaningful because both modes solve the same
generated tasks.

In standard mode, the model sees the public task spec and native tools. For
`hidden_dag`, those tools are generated functions. For `repair_chain`, they are
operations such as `scan_object`, `inspect_tool`, `move_to`, `apply_tool`, and
`test_object`. The model calls tools directly and eventually calls `submit`.

In PTC mode, the model sees a Python tool. A persistent workspace is mounted at
`/workspace`. The workspace contains a small package:

```python
from ptc import task, tools, finish
```

The PTC model can inspect the public spec:

```python
task.spec()
```

It can list and call primitive tools:

```python
tools.list()
tools.call("scan_object", object_id="machine_0")
tools.scan_object(object_id="machine_0")
```

It finishes by submitting an answer:

```python
finish({"repaired": ["machine_0", "machine_1"]})
```

The important constraint is that the `ptc` package is only a public interface.
It does not expose hidden state or reward. It routes calls to the same host-side
environment used by standard mode. The distinction is not privileged
information; it is the ability to construct durable machinery around the public
tools.

That is the research object.

## Task Family I: Hidden DAG

`hidden_dag` is the cleanest family. It is closest to FuncBenchGen and serves as
a bridge from ordinary function-calling evaluation into Undertow.

The generator creates a set of deterministic functions over integer variables.
The public spec includes:

- initial variable values,
- a target variable,
- generated function names,
- each function's input variables,
- each function's output variable,
- JSON tool schemas.

A small public spec might say:

```json
{
  "family": "hidden_dag",
  "initial_values": {"v0": 964, "source_5": 494},
  "target": "v5",
  "functions": [
    {
      "name": "core_0",
      "inputs": ["v0"],
      "output": "v1",
      "description": "Transforms v0 into v1."
    },
    {
      "name": "core_4",
      "inputs": ["v4", "v6", "v8"],
      "output": "v5",
      "description": "Combines v4, v6, v8 into v5."
    }
  ]
}
```

The hidden state contains the true value table and the ground-truth solution
order. The model does not see that table. It must produce values by calling
functions with correct already-known inputs.

A successful call returns:

```json
{"ok": true, "variable": "v1", "value": 6761}
```

The main challenge is not schema syntax. The schema is straightforward. The
challenge is dataflow discipline:

- call only functions whose inputs are known,
- extract the returned `value`,
- store it under the returned `variable`,
- avoid connected distractors,
- submit the target value only after computing it.

The difficulty knobs are:

- `n_core`: the number of required functions.
- `depth`: the longest required dependency path.
- `n_connected_distractors`: irrelevant functions connected to known or core
variables.
- `n_disconnected_distractors`: irrelevant functions outside the solution path.
- `observation_mode`: whether the environment returns only the new observation
or reminds the model of known values.

Why is this interesting for PTC?

Because the ideal PTC strategy is almost embarrassingly simple, and that is the
point. A PTC agent can parse the public function list, maintain a `known_values`
dictionary, repeatedly find functions whose inputs are available, call them, and
cache outputs. This is not a deep algorithm. It is a small topological executor.
But writing that executor turns a multi-turn memory problem into a local state
management problem.

Standard tool-calling agents can solve these tasks too, especially when the DAG
is small. But they are more exposed to context-level slips: using the whole
observation object as an integer, calling a child before its parent, forgetting a
returned value, or substituting an earlier value with the same-looking name.

`hidden_dag` therefore tests whether PTC helps with controlled dataflow tracking.
It is the benchmark's accounting ledger: exact, dry, and unforgiving.

## Task Family II: Repair Chain

`repair_chain` is the more stateful family and the better test of Undertow's
purpose.

The world contains broken machines, tools, locations, and hidden repair stages.
Each machine has fields like:

```json
{
  "id": "machine_0",
  "status": "broken",
  "part": "part_0",
  "location": "bay_1",
  "repair_stage": 0,
  "observation_version": 0
}
```

The public spec lists machine ids, bay ids, available tool ids, and operation
schemas. The model can call:

- `scan_object(object_id)`: observe a machine's current status, part, location,
repair stage, and observation version.
- `inspect_tool(tool_id)`: observe a tool's kind, compatible parts, and the
stages where it can repair each part.
- `move_to(location)`: move the agent to a bay.
- `apply_tool(tool_id, object_id, observation_version)`: apply a tool at the
current location.
- `test_object(object_id, observation_version)`: test whether a machine is
repaired.

The tool inspection response is what makes the task publicly inferable:

```json
{
  "ok": true,
  "tool_id": "tool_3",
  "kind": "repair",
  "compatible_parts": ["part_0", "part_2"],
  "compatible_stages_by_part": {
    "part_0": [1],
    "part_2": [0, 2]
  }
}
```

A good solver can combine scans and inspections into a repair plan:

1. inspect tools once,
2. scan each machine,
3. build a map from `(part, stage)` to `tool_id`,
4. move to the machine location,
5. apply the stage-appropriate tools in order,
6. update observation versions after each state-changing action,
7. test the machine,
8. submit the repaired object ids.

The difficulty knobs are:

- `n_objects`: how many machines must be repaired.
- `n_repair_steps`: how many sequential stages each machine requires.
- `n_tools`: how many real tools are available.
- `n_distractor_tools`: how many tools are present but useless.
- `staleness_rate`: whether object observations become stale after repair
actions.
- `partial_observation_rate`: whether scans hide fields such as part or
location.

Why is this interesting for PTC?

Because it is no longer just a DAG of values. It is a mutable state machine.

A standard tool-calling model has to keep several tables in its head:

- which machine is in which bay,
- which machine has which part,
- which tool repairs which stage for that part,
- what the current repair stage is,
- what the current observation version is,
- where the agent is currently located,
- which machines have already been tested.

Each successful repair action changes state. If observations become stale, an
old `observation_version` can no longer be safely reused. A model that fails to
update this local view can make calls that are superficially reasonable but
semantically invalid.

The PTC version gives the model a better affordance. It can write a small repair
ledger:

```python
machines = {}
stage_tool = {}
current_location = None
```

It can log every call. It can assert that a stage exists before applying a tool.
It can update the version after every successful `apply_tool`. It can group
machines by location to avoid unnecessary moves. It can stop retrying a tool
after a constraint violation.

This is the kind of work PTC is supposed to make possible. The benchmark is not
checking whether Python itself is magic. It is checking whether the model uses a
programmable workspace to become a more reliable agent.

## Metrics and Failure Taxonomy

Undertow reports both success and the shape of the attempt.

The core metrics are:

- `success`: whether the final submitted answer satisfies the reward checker.
- `num_calls`: consumed tool-call budget.
- `budget`: total budget.
- `remaining_budget`: unused budget at the end.
- `budget_used_ratio`: consumed calls divided by budget.
- `progress_score`: family-specific normalized partial progress.
- `useful_calls`: consumed calls with a non-empty state delta.
- `wasted_calls`: consumed calls that produced a failure code.
- `failure_counter`: counts by machine-checked failure code.
- `first_failure`: first failure code in the trace.
- `final_failure`: final failure code for failed episodes.

The failure codes are intentionally mechanical:

- `invalid_action`: the tool does not exist or the episode is already closed.
- `schema_error`: arguments do not match the schema.
- `unknown_state_reference`: the agent referenced an unknown object, tool, or
variable.
- `incorrect_value`: the agent supplied a known variable name with the wrong
value.
- `stale_state_reference`: the agent used an obsolete observation version.
- `violated_constraint`: the action was syntactically valid but violated world
constraints.
- `distractor_action`: the agent used a distractor tool or irrelevant function.
- `premature_submit`: the submitted answer did not satisfy the target.
- `budget_exceeded`: the agent tried to continue after consuming its budget.
- `repeated_call`: the agent repeated an identical call.

This taxonomy matters because success rate alone is too blunt. Two models can
both fail, but one may fail after making real progress while the other never
establishes the first piece of state. One may be defeated by connected
distractors. Another may compute almost everything but submit the wrong answer.
Those are different failures, and a useful benchmark should keep them separate.

## The Experiment

The first experiment has one purpose: compare standard and PTC interaction modes
under the same generated tasks.

The fixed grid is stored in:

```text
experiments/undertow_blog/configs.json
```

The experiment runner writes normalized raw rows to:

```text
experiments/undertow_blog/results/raw/*.jsonl
```

The aggregator writes:

```text
experiments/undertow_blog/results/summary.csv
experiments/undertow_blog/results/by_config.csv
experiments/undertow_blog/results/failures.csv
```

To preview the commands without running model calls:

```bash
uv run python experiments/undertow_blog/run.py \
  --model openai/gpt-5.5 \
  --limit 2 \
  --dry-run
```

To run the full experiment for GPT-5.5:

```bash
uv run python experiments/undertow_blog/run.py \
  --model openai/gpt-5.5
```

To run it for GPT-5.2:

```bash
uv run python experiments/undertow_blog/run.py \
  --model openai/gpt-5.2
```

To aggregate after the runs finish:

```bash
uv run python experiments/undertow_blog/aggregate.py
```

If the full grid is too expensive for a first pass, run a small smoke subset:

```bash
uv run python experiments/undertow_blog/run.py \
  --model openai/gpt-5.5 \
  --limit 8

uv run python experiments/undertow_blog/aggregate.py
```

After those files exist, the next step is to inspect the summary tables rather
than speculate. The analysis should answer:

- Does PTC improve success rate overall?
- Is the PTC advantage larger on `repair_chain` than `hidden_dag`?
- Does PTC reduce budget exhaustion?
- Does PTC produce a higher useful-call rate?
- Do standard failures cluster around stale state, constraint violations, and
repeated calls?
- Do PTC failures cluster around over-exploration, budget use, or local solver
mistakes?

Those are the claims the benchmark is designed to make measurable.

## What I Expect Before Seeing Results

The expected result is not that PTC wins everywhere.

On small `hidden_dag` tasks, standard tool calling may be competitive. The task
is clean, the dependency graph is visible in the public spec, and a strong model
can often walk the sequence directly. PTC can still help, but it may spend extra
turns writing scaffolding for a problem that does not need much scaffolding.

On deeper or more distractor-heavy `hidden_dag` tasks, PTC should begin to look
better if the model writes even a modest executor. Its advantage should come from
stable value propagation and avoiding repeated reasoning over the same public
spec.

On `repair_chain`, PTC should be more naturally useful. The environment asks for
a persistent table of objects, versions, locations, parts, tools, and stages.
That is exactly the sort of structure a Python workspace can hold. If PTC does
not improve there, the traces should be revealing: either the model failed to
build the obvious ledger, overbuilt until the budget disappeared, or misunderstood
the public stage information.

The negative result would also be interesting. If standard mode matches PTC on
these stateful tasks, it suggests either the tasks are still too small, the public
observations are too generous, or the current PTC prompting fails to elicit the
control layer it is meant to test. Undertow is useful in either case because it
makes that diagnosis traceable.

## What Undertow Is Not

Undertow is not a claim that synthetic tasks are sufficient for agent evaluation.
They are not.

It is not a claim that Python workspaces should be available to every deployed
agent. That is an engineering question.

It is not a claim that PTC always beats standard tool calling. It should not.

Undertow is narrower and, for that reason, more interpretable. It asks when the
ability to build local machinery changes the reliability of tool use. The
benchmark is synthetic so that the answer is not hidden behind task collection
noise. It is controllable so that difficulty can be moved one knob at a time. It
is stateful so that the agent must do more than choose functions. It is traced so
that failure analysis can be mechanical rather than anecdotal.

That is the standard to hold it to.

## Closing

The old habit in tool-use benchmarks is to count whether the model reached the
answer. Undertow keeps that count, but it is more interested in the account book:
what the agent knew, what it changed, what it repeated, what it used too early,
and what it failed to update.

Programmatic tool calling is only interesting if the program becomes a working
instrument: a ledger, a planner, a verifier, a memory. Undertow is a small
benchmark for seeing whether that happens.

Run the experiment grid, aggregate the tables, and then the traces can speak.