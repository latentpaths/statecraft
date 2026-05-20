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

Undertow reports one scalar reward and a small diagnostic surface.

The core metrics are:

- `reward`: the single benchmark score, combining exact completion, partial
progress, and call efficiency.
- `solved`: whether the final submitted answer satisfies the exact target.
- `calls`: consumed tool-call budget.
- `budget`: total budget.
- `reference_calls`: the public-solver call count used as the efficiency
reference.
- `efficiency`: `reference_calls / max(calls, reference_calls)`.
- `progress`: family-specific normalized partial progress.
- `state_changes`: consumed calls with a non-empty state delta.
- `failure_calls`: consumed calls that produced a failure code.
- `failure_counter`: counts by machine-checked failure code.
- `first_failure`: first failure code in the trace.
- `final_failure`: final failure code for failed episodes.

The reward formula is:

```text
if solved:
    reward = 0.8 + 0.2 * efficiency
else:
    reward = 0.5 * progress * efficiency
```

`state_changes` is deliberately conservative. A scan, inspection, or early test
can be useful even when it does not change hidden state. `failure_calls` is also
diagnostic rather than moral language: it counts failure-coded calls without
claiming that every such call was wasted.

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

This taxonomy matters because solve rate alone is too blunt. Two models can
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

The commands are deliberately plain:

```bash
uv run python experiments/undertow_blog/run.py \
  --model qwen/qwen3.6-35b-a3b

uv run python experiments/undertow_blog/run.py \
  --model z-ai/glm-5

uv run python experiments/undertow_blog/aggregate.py
```

The current result set contains 307 normalized episodes. It covers both
interaction modes for Qwen3.6-35B-A3B and GLM-5. Most cells contain 40 episodes:
eight generator configurations times five seeds. One PTC `repair_chain` slice
for GLM-5 contains 27 episodes because that run set is incomplete, so its
repair-chain aggregate should be read as directional rather than final.

## Calibration: Why the Budget Changed

The first budget policy was too severe. It looked principled on paper but failed
as an evaluation instrument. In early runs, models reached high partial progress
and then lost purely because the budget admitted almost no discovery.

The clearest example came from `hidden_dag` with five core functions and no
connected distractors. The budget was 10 calls. Standard agents often reached
`progress = 0.8`, meaning four of five required values had been produced,
then hit `budget_exceeded`. That is not the failure the benchmark is meant to
study. It says only that the budget fit a near-oracle trace, not that the agent
failed to build or follow a plan.

The policy was changed to:

```text
budget = 3 * public_solver_call_count
```

For `hidden_dag`, the public-solver count is all generated functions plus
submit. For `repair_chain`, it is inspect all tools, scan all objects, move,
apply all repair stages, test all objects, and submit. This keeps budget finite
and meaningful, but gives room for the behavior real agents actually exhibit:
discovery, a repeated scan, a wrong hypothesis, or a premature test.

That change is part of the benchmark, not a cosmetic adjustment. A benchmark
that collapses when the model takes one exploratory step is measuring thrift, not
agent control.

## Results

The first completed run should be read as a diagnostic run, not as the final
standard-vs-PTC comparison.

That distinction matters. The trajectories exposed a benchmark bug in the
standard harness: standard mode reconstructed the environment from a serialized
snapshot between native tool calls, but the snapshot preserved only generic
episode bookkeeping. It did not preserve family-specific hidden mutable state:
`hidden_dag`'s known value table, or `repair_chain`'s current location and
machine repair stages.

PTC mostly avoided this failure because its RPC-backed primitive tools were
served by a live session. Standard mode did not. A successful standard
`hidden_dag` call could compute `v1`, but the next call would restore an
environment that had forgotten `v1`. A successful standard `move_to("bay_0")`
could be followed by `apply_tool(...)` in an environment whose
`current_location` was again `null`.

This is exactly why Undertow stores traces. The aggregate numbers looked like a
large PTC advantage. The trajectories showed that part of the gap was real model
behavior, and part of it was a state-restore bug in the standard adapter. That
bug has been fixed by making each family serialize its own mutable hidden state
inside the episode snapshot. The numbers below are therefore retained as a
pre-fix diagnostic, not as the clean result table.

With that caveat, the pre-fix run was still informative.

| mode | family | model | episodes | solve rate | avg calls | failed progress | budget exhaustion | state-change rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PTC | hidden_dag | Qwen3.6-35B-A3B | 40 | 95.0% | 13.25 | 0.40 | 0.0% | 80.4% |
| PTC | hidden_dag | GLM-5 | 40 | 97.5% | 11.68 | 0.20 | 0.0% | 83.9% |
| standard | hidden_dag | Qwen3.6-35B-A3B | 40 | 0.0% | 21.43 | 0.625 | 40.0% | 59.9% |
| standard | hidden_dag | GLM-5 | 40 | 0.0% | 32.58 | 0.625 | 85.0% | 60.4% |

On `hidden_dag`, the PTC agents did what the benchmark hoped they would do. They
used the workspace as a value ledger and execution surface. They parsed the
public function list, built an explicit known-value table, called functions when
their inputs were available, extracted returned `{"variable", "value"}` pairs,
and submitted the final value. The best PTC traces read less like improvised
chat and more like a small topological executor.

The standard traces looked different. Models made real progress, but they often
reacted to a missing reference by trying to refresh the world: repeating earlier
function calls, reusing partial argument sets, or experimenting with empty
schemas. Some of this is model behavior. In the pre-fix run, some of it was also
caused by the restore bug: the environment had forgotten values the trace had
already produced.

The failure counters make the distinction concrete. For standard `hidden_dag`,
the dominant errors were repeated calls and unknown state references. Qwen's
standard runs produced 311 repeated-call failures and 305 unknown-reference
failures. GLM-5's standard runs produced 572 repeated-call failures and 468
unknown-reference failures. These are precisely the failures a small local
`known_values` dictionary is supposed to prevent.

PTC was not error-free. It still produced repeated calls and some unknown
references. But those errors did not usually derail the episode. The workspace
made recovery cheap enough and state explicit enough that the final target value
was still submitted correctly.

The `repair_chain` result is more subtle and more interesting, but the same
pre-fix caveat applies even more strongly.

| mode | family | model | episodes | solve rate | avg calls | failed progress | budget exhaustion | state-change rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PTC | repair_chain | Qwen3.6-35B-A3B | 40 | 0.0% | 24.80 | 1.00 | 0.0% | 28.8% |
| PTC | repair_chain | GLM-5 | 27 | 0.0% | 20.44 | 1.00 | 0.0% | 26.3% |
| standard | repair_chain | Qwen3.6-35B-A3B | 40 | 0.0% | 41.95 | 0.00 | 45.0% | 43.8% |
| standard | repair_chain | GLM-5 | 40 | 0.0% | 53.75 | 0.00 | 77.5% | 28.4% |

If one only reads exact completion, `repair_chain` looks like a failure for
everyone. That is technically true under `solved`. But it is not the whole story.

PTC reached `progress = 1.0` on average. That means the PTC agents
generally completed the repair state transitions even though final reward stayed
at the old binary value of zero. They inspected tools, built stage maps from public inspection data, moved
to the right bay, applied the stage-appropriate tool sequence, updated
observation versions, and reached repaired states.

A sampled PTC trace shows the pattern plainly. The agent inspected tools,
identified the stage sequence, moved to the bay, applied the full repair chain,
scanned the machine at the repaired stage, and received a successful final test.
Then it called:

```python
finish(answer="All critical machines have been repaired...")
```

The environment expected structured repaired object ids:

```python
finish({"repaired": ["machine_0"]})
```

So exact completion stayed false and the failure was classified as
`premature_submit`. Under the cleaned metric surface, this kind of run receives
partial `reward` through progress and efficiency, while `solved` remains false.

This is a valuable failure. It says PTC succeeded at the stateful control problem
and failed at final answer packaging. That is not a minor nuisance, because
interfaces matter. But it is very different from failing to repair the machine.
The trace-level metrics make that difference visible.

In standard mode, the pre-fix `repair_chain` traces were dominated by repeated
calls and violated constraints. Qwen standard runs produced 871 repeated-call
failures and 341 constraint violations. GLM-5 standard runs produced 880
repeated-call failures and 666 constraint violations. Before the restore fix,
these numbers should not be interpreted as a pure model failure. A representative
trace showed a successful `move_to`, followed by an `apply_tool` response whose
`current_location` was `null`. The agent was being judged inside a world that had
lost the movement state between calls.

PTC repair-chain failures were different. They were dominated by repeated calls
and premature submit, with much lower constraint-violation pressure. The agent
was still inefficient, but it was inefficient around a working plan rather than
lost before the plan.

## Analysis

The trajectory audit changes the interpretation in an important way.

The cleanest claim from the PTC traces is still that PTC helps when the task can
be turned into a small local state machine.

On `hidden_dag`, the local state machine is a table of known values and a queue
of callable functions. That is enough to turn a brittle multi-turn reasoning task
into a deterministic execution loop. The PTC solve rates in the pre-fix run
were correspondingly strong: 95.0% and 97.5%.

On `repair_chain`, the local state machine is a repair ledger: machine, part,
location, stage, observation version, stage-to-tool map, and repaired status. PTC
appears to build enough of that ledger to complete the hidden state transition.
The exact completion check remains false because the last action is not yet
disciplined. The model solves the world and then writes the wrong receipt.

This is exactly why Undertow records `solved`, `progress`, and `efficiency`
separately. `reward` is the single scalar benchmark score. `solved` is the exact
completion check. `progress` is the audit trail. In a deployment setting, the
final answer-shape error would still matter. In a benchmark design setting, it
tells us where to look next: not at the repair transition model, but at the
finish contract.

The standard results must be rerun after the snapshot fix before making the
stronger comparative claim. The pre-fix traces still revealed useful model
behavior: standard agents tended to repair uncertainty by repeating tools rather
than by maintaining an explicit ledger. But the old standard solve rates and
repair progress are not publication-grade because the harness itself was
forgetting hidden mutable state.

So the current conclusion is not simply "PTC gets more reward." It is more
precise and more conservative:

1. PTC converts hidden DAG tasks into high-reliability executions in the current
   run set.
2. PTC substantially improves stateful repair progress.
3. PTC repair agents currently fail at structured final submission, not at the
   underlying repair transition.
4. The initial standard results exposed a real benchmark bug: family-specific
   hidden state was not serialized across native tool calls.
5. The standard comparison should be rerun after the snapshot fix, and the
   repaired results should replace the diagnostic table above.

That is a useful benchmark outcome. It exposes the promise of programmatic tool
calling, the interface brittleness of final submission, and the value of
trace-level auditing for catching benchmark infrastructure errors before they
become claims.

## What To Fix Next

The first fix was not to make the benchmark easier. It was to make the comparison
valid: environment snapshots now include family-specific mutable state.

The next fix should also be narrow. The reward is correct to require a
structured final answer. Instead of loosening reward, the PTC prompt and
workspace should make the finish contract harder to miss.

For example, the task prompt can say:

```python
finish({"repaired": ["machine_0", "machine_1"]})
```

for `repair_chain`, and:

```python
finish({"value": 1234})
```

for `hidden_dag`.

That is not giving away the solution. It is specifying the output protocol. The
current traces show that models can complete the hidden state transition and
still fail because they submit prose. A benchmark should punish that, but the
paper should report it honestly as an answer-protocol failure rather than a
repair-planning failure.

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

The traces have now spoken clearly enough for a first conclusion: PTC is already
solving the dataflow task and largely solving the state-transition part of the
repair task. They also exposed a standard-harness restore bug, which is exactly
the kind of flaw a trace-first benchmark should catch before publication. The
remaining work is to rerun the clean comparison after that fix and tighten the
final contract so that the benchmark separates "could not solve the world" from
"solved the world but wrote the receipt incorrectly."
