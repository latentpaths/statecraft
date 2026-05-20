Undertow: A Benchmark for Programmatic Tool Calling in Hidden Stateful Environments

TL;DR

Undertow is a small, synthetic benchmark for testing a specific claim: programmatic tool calling should matter when tool use is no longer a sequence of isolated API calls, but a problem of maintaining state, building memory, verifying progress, and controlling a partially observed environment.

The early diagnostic run suggests three things.

First, PTC agents already handle the hidden dataflow task well. In hidden_dag, both tested PTC models built the obvious but important machinery: a local value table, a callable-function queue, and a simple execution loop. They solved the task at high reliability.

Second, PTC agents made substantial progress on the harder stateful repair task. In repair_chain, they generally completed the hidden repair transitions, but often failed final reward by submitting prose instead of the required structured answer. That is not a trivial error, because interfaces matter, but it is different from failing to repair the machines.

Third, the first standard-vs-PTC comparison exposed a benchmark bug. The standard harness was restoring environments between native tool calls without preserving family-specific mutable hidden state. That made part of the apparent PTC advantage invalid as a clean comparison. The bug has since been fixed, and the standard comparison should be rerun before making a stronger claim.

That is the central lesson so far: Undertow is doing what a trace-first benchmark should do. It shows where PTC appears genuinely useful, where models fail at the interface boundary, and where the benchmark infrastructure itself needs repair before the results become claims.

⸻

1. The Question Undertow Is Built To Ask

There is a narrow but important question hiding inside the larger conversation about agents.

Can a model call tools? Can a model write code? I think these questions have been firmly settlted once we look at the current benchmarks.

The more interesting question is whether a model, given a programmable workspace, can build the missing control layer around tools: memory, state tracking, planning, verification, retry logic, and enough local abstraction to keep a partially observed environment straight.

That is the purpose of Undertow.

Undertow is a synthetic, controllable benchmark for comparing ordinary tool-calling agents with programmatic tool-calling agents. It asks models to interact with hidden environments through primitive public tools. The environment has state. Observations are partial. Actions can go stale. Some tools are irrelevant. The target condition is hidden behind a reward checker. The model does not get to inspect the answer, and local checks are useful only as discipline, not as authority.

The benchmark rests on a simple bet: programmatic tool calling should not matter much when a task is merely a short list of obvious calls. It should matter when the task requires durable cognition.

A chat agent can often manage only so many calls efficicently. It can read a schema, call a function, observe the result, and answer. But real agent work usually fails later. It fails when the agent must remember what it learned, avoid reusing stale facts, distinguish useful actions from distractors, preserve intermediate state, and recover after an attempted action does not do what it expected.

Undertow is designed around that later failure.

⸻

1. Why this Tool-Use Benchmark?

Most tool-use evaluations measure whether a model can select and call APIs. That is a legitimate capability, but it is not the whole agent problem.

A serious agent does not merely call a tool. It keeps accounts.

It must know what has already been tried, which observations are still valid, what state has changed, what information is missing, what action is safe to take next, and whether the apparent solution actually satisfies the external world. The failure mode is not always ignorance. Often the model had the right information once, then lost it.

FuncBenchGen gives a clean example of how to study part of this space. It constructs synthetic, contamination-free function-calling tasks from hidden dependency DAGs. The model sees a set of functions, input variables, and a target variable. It must infer the call sequence needed to compute the target. The important feature of FuncBenchGen is not only the DAG. It is the experimental discipline: generated tasks, controlled difficulty, hidden ground truth, machine-checked traces, and a failure taxonomy.

Undertow follows that spirit, but changes the center of gravity.

FuncBenchGen is mostly about hidden dataflow. Undertow is about hidden state.

It keeps the contamination-free and controllable construction, but adds the ingredient that makes programmatic tool calling worth testing: a workspace in which the agent can build its own operating layer around the public tools.

In standard mode, the model sees native tools and must solve the task through the ordinary conversation/tool-call loop.

In PTC mode, the model sees Python. It can write files, build tables, cache observations, construct a planner, run local checks, and call the same primitive environment tools through a small ptc package. The hidden state remains hidden. The reward remains protected. But the agent’s working memory no longer has to live entirely inside a transient conversation.

That is the distinction Undertow is built to isolate.

⸻

1. The Shape of the Benchmark

Undertow v0 deliberately stays small. It has two task families:

1. hidden_dag: deterministic hidden function-dependency tasks.
2. repair_chain: mutable, partially observable repair tasks.

It has two interaction modes:

1. standard: the model receives the native tools directly.
2. ptc: the model receives a Python workspace and calls the tools inside of the Python workspace.

Undertow’s role is restrained: hold the environment fixed, control the difficulty knobs, and ask whether a programmable workspace changes reliability.

The initial experiment compares standard and PTC agents across both task families with a fixed generated grid:

hidden_dag:
  n_core in {5, 10}
  depth in {2, 5}
  n_connected_distractors in {0, 5}
repair_chain:
  n_objects in {1, 3}
  n_repair_steps in {2, 3}
  n_distractor_tools in {0, 2}
seeds:
  0..4
modes:
  standard, ptc

For both families, the default budget is three times the public-solver call count.

In hidden_dag, the public-solver count is all generated functions plus submit. In repair_chain, it is inspecting every tool once, scanning every object once, moving once per object, applying every required repair stage, testing every object, and submitting.

The extra budget is intentional. A real model will do discovery. It may repeat a scan, make an occasional wrong hypothesis, test too early, or retry after a confusing observation. The benchmark should measure those choices, not collapse immediately because the budget only fits an oracle-perfect trace.

This is not an exhaustive sweep. It is the first clean cut: small enough to inspect, large enough to expose failure modes, and structured enough to support a real comparison.

⸻

1. Formalization

An Undertow task is a tuple:

τ = (S₀, Ω, A, T, R, B)

where:

S₀ = hidden initial state
Ω  = observation functions
A  = state-changing actions
T  = target condition
R  = reward checker
B  = tool-call budget

The agent never observes S₀ directly. It receives only a public task specification: tool schemas, public identifiers, controllable task parameters, and a target description.

At step t, the agent chooses an operation:

aₜ ∈ Ω ∪ A

The environment applies the transition:

Sₜ₊₁ = transition(Sₜ, aₜ)

and returns a public observation:

oₜ = observe(Sₜ₊₁, aₜ)

The episode ends when the agent submits an answer or exhausts the budget. Success is defined only by the reward checker:

R(S_T, answer) = 1

This matters. The model can maintain a local verifier. It can write a plan. It can declare that the task is complete. None of that is authoritative. The environment decides.

Every call is recorded as a trace event:

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

The trace is the basis for scoring and analysis. It lets the benchmark classify failures without human annotation: incorrect value, unknown reference, stale observation, violated constraint, distractor action, repeated call, premature submit, and budget exhaustion.

This is important because success rate alone is too blunt. Two agents can both fail, but one may fail after completing the hidden state transition while the other never established the first piece of state.

⸻

1. The Two Interaction Modes

The comparison is meaningful only because both modes solve the same generated tasks.

Standard Mode

In standard mode, the model sees the public task spec and native tools. For hidden_dag, those tools are generated functions. For repair_chain, they are operations such as:

scan_object
inspect_tool
move_to
apply_tool
test_object
submit

The model calls tools directly through the usual conversation/tool-call loop and eventually calls submit.

PTC Mode

In PTC mode, the model sees a Python workspace. A persistent directory is mounted at:

/workspace

The workspace contains a small package:

from ptc import task, tools, finish

The PTC agent can inspect the public spec:

task.spec()

It can list and call primitive tools:

tools.list()
tools.call("scan_object", object_id="machine_0")
tools.scan_object(object_id="machine_0")

It finishes by submitting an answer:

finish({"repaired": ["machine_0", "machine_1"]})

The constraint is strict: the ptc package is only a public interface. It does not expose hidden state. It does not expose reward. It routes calls to the same host-side environment used by standard mode.

The distinction is not privileged information.

The distinction is that the PTC agent can construct durable machinery around the public tools: ledgers, parsers, caches, verifiers, planners, retry loops, and state trackers.

⸻

1. Task Family I: hidden_dag

hidden_dag is the cleanest family. It is closest to FuncBenchGen and serves as a bridge from ordinary function-calling evaluation into Undertow.

The generator creates a set of deterministic functions over integer variables. The public spec includes:

initial variable values
target variable
generated function names
each function's input variables
each function's output variable
JSON tool schemas

A small public spec might look like this:

{
  "family": "hidden_dag",
  "initial_values": {
    "v0": 964,
    "source_5": 494
  },
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

The hidden state contains the true value table and the ground-truth solution order. The model does not see that table. It must produce values by calling functions with correct already-known inputs.

A successful function call returns:

{
  "ok": true,
  "variable": "v1",
  "value": 6761
}

The main challenge is not schema syntax. The schemas are straightforward. The challenge is dataflow discipline:

call only functions whose inputs are known
extract the returned value
store it under the returned variable
avoid connected distractors
submit the target value only after computing it

The difficulty knobs are:

n_core: number of required functions
depth: longest required dependency path
n_connected_distractors: irrelevant functions connected to known or core variables
n_disconnected_distractors: irrelevant functions outside the solution path
observation_mode: whether the environment returns only the new observation or reminds the model of known values

Why is this interesting for PTC?

Because the ideal PTC strategy is almost embarrassingly simple, and that is the point. A PTC agent can parse the public function list, maintain a known_values dictionary, repeatedly find functions whose inputs are available, call them, and cache outputs.

This is not a deep algorithm. It is a small topological executor. But writing that executor turns a multi-turn memory problem into a local state-management problem.

Standard tool-calling agents can solve these tasks too, especially when the DAG is small. But they are more exposed to context-level slips: using the whole observation object as an integer, calling a child before its parent, forgetting a returned value, or substituting an earlier value with a similar name.

hidden_dag is therefore Undertow’s accounting ledger: exact, dry, and unforgiving. It tests whether PTC helps with controlled dataflow tracking.

⸻

1. Task Family II: repair_chain

repair_chain is the more stateful family and the better test of Undertow’s purpose.

The world contains broken machines, tools, locations, and hidden repair stages. Each machine has fields like:

{
  "id": "machine_0",
  "status": "broken",
  "part": "part_0",
  "location": "bay_1",
  "repair_stage": 0,
  "observation_version": 0
}

The public spec lists machine ids, bay ids, available tool ids, and operation schemas. The model can call:

scan_object(object_id)
inspect_tool(tool_id)
move_to(location)
apply_tool(tool_id, object_id, observation_version)
test_object(object_id, observation_version)

The tool inspection response is what makes the task publicly inferable:

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

A good solver can combine scans and inspections into a repair plan:

1. inspect tools once
2. scan each machine
3. build a map from (part, stage) to tool_id
4. move to the machine location
5. apply the stage-appropriate tools in order
6. update observation versions after each state-changing action
7. test the machine
8. submit the repaired object ids

The difficulty knobs are:

n_objects: number of machines to repair
n_repair_steps: number of sequential repair stages per machine
n_tools: number of real tools
n_distractor_tools: number of useless tools
staleness_rate: whether object observations become stale after repair actions
partial_observation_rate: whether scans hide fields such as part or location

Why is this interesting for PTC?

Because it is no longer just a DAG of values. It is a mutable state machine.

A standard tool-calling model has to keep several tables in its head:

which machine is in which bay
which machine has which part
which tool repairs which stage for that part
what the current repair stage is
what the current observation version is
where the agent is currently located
which machines have already been tested

Each successful repair action changes state. If observations become stale, an old observation_version can no longer be safely reused. A model that fails to update its local view can make calls that are syntactically reasonable but semantically invalid.

The PTC version gives the model a better affordance. It can write a small repair ledger:

machines = {}
stage_tool = {}
current_location = None

It can log every call. It can assert that a stage exists before applying a tool. It can update the observation version after every successful apply_tool. It can group machines by location to avoid unnecessary moves. It can stop retrying a tool after a constraint violation.

This is the kind of work PTC is supposed to make possible. The benchmark is not checking whether Python is magic. It is checking whether the model uses a programmable workspace to become a more reliable agent.

⸻

1. Metrics and Failure Taxonomy

Undertow reports both success and the shape of the attempt.

The core metrics are:

success: whether the final submitted answer satisfies the reward checker
num_calls: consumed tool-call budget
budget: total budget
remaining_budget: unused budget at the end
budget_used_ratio: consumed calls divided by budget
progress_score: family-specific normalized partial progress
useful_calls: consumed calls with a non-empty state delta
wasted_calls: consumed calls that produced a failure code
failure_counter: counts by machine-checked failure code
first_failure: first failure code in the trace
final_failure: final failure code for failed episodes

The failure codes are intentionally mechanical:

invalid_action: tool does not exist or episode is already closed
schema_error: arguments do not match the schema
unknown_state_reference: agent referenced an unknown object, tool, or variable
incorrect_value: agent supplied a known variable name with the wrong value
stale_state_reference: agent used an obsolete observation version
violated_constraint: action was syntactically valid but violated world constraints
distractor_action: agent used a distractor tool or irrelevant function
premature_submit: submitted answer did not satisfy the target
budget_exceeded: agent tried to continue after consuming its budget
repeated_call: agent repeated an identical call

This taxonomy matters because failure has structure.

One model may fail because it never discovers the right tool. Another may discover the tool, repair the object, test it successfully, and then submit the wrong answer shape. A third may know the right action but use a stale observation version. These are not the same failure, and we should not collapse them into the same number.

![Failure Taxonomy](results/figures/03_failure_taxonomy.png)

⸻

1. The First Experiment

The first experiment has one purpose: compare standard and PTC interaction modes under the same generated tasks.

The fixed grid is stored in:

experiments/undertow_blog/configs.json

The experiment runner writes normalized raw rows to:

experiments/undertow_blog/results/raw/*.jsonl

The aggregator writes:

experiments/undertow_blog/results/summary.csv
experiments/undertow_blog/results/by_config.csv
experiments/undertow_blog/results/failures.csv

The commands are deliberately plain:

uv run python experiments/undertow_blog/run.py   
  --model qwen/qwen3.6-35b-a3b
uv run python experiments/undertow_blog/run.py   
  --model z-ai/glm-5
uv run python experiments/undertow_blog/aggregate.py

The current result set contains 307 normalized episodes. It covers both interaction modes for Qwen3.6-35B-A3B and GLM-5. Most cells contain 40 episodes: eight generator configurations times five seeds. One PTC repair_chain slice for GLM-5 contains 27 episodes because that run set is incomplete, so its repair-chain aggregate should be read as directional rather than final.

⸻

1. Calibration: Why the Budget Changed

The first budget policy was too severe.

It looked principled on paper, but failed as an evaluation instrument. In early runs, models reached high partial progress and then lost purely because the budget admitted almost no discovery.

The clearest example came from hidden_dag with five core functions and no connected distractors. The budget was 10 calls. Standard agents often reached:

progress_score = 0.8

meaning four of five required values had been produced, then hit budget_exceeded.

That is not the failure Undertow is meant to study. It says only that the budget fit a near-oracle trace. It does not tell us whether the agent failed to build or follow a plan.

The policy was changed to:

budget = 3 × public_solver_call_count

For hidden_dag, the public-solver count is all generated functions plus submit. For repair_chain, it is inspecting all tools, scanning all objects, moving, applying all repair stages, testing all objects, and submitting.

This keeps the budget finite and meaningful, while giving room for the behavior real agents actually exhibit: discovery, repeated scans, wrong hypotheses, premature tests, and occasional retries.

![Budget Exhaustion](results/figures/08_budget_exhaustion.png)

⸻

1. Results

The trajectories exposed a benchmark bug in the standard harness. Standard mode reconstructed the environment from a serialized snapshot between native tool calls, but the snapshot preserved only generic episode bookkeeping. It did not preserve family-specific hidden mutable state: hidden_dag’s known value table, or repair_chain’s current location and machine repair stages.

PTC mostly avoided this failure because its RPC-backed primitive tools were served by a live session. Standard mode did not.

A successful standard hidden_dag call could compute v1, but the next call would restore an environment that had forgotten v1. A successful standard move_to("bay_0") could be followed by apply_tool(...) in an environment whose current_location was again null.

The aggregate numbers looked like a large PTC advantage. The trajectories showed that part of the gap was real model behavior, and part of it was a state-restore bug in the standard environemnt implementation.

That bug has now been fixed by making each family serialize its own mutable hidden state inside the episode snapshot. The numbers below are therefore retained as a pre-fix diagnostic, not as a clean result table.

![Headline Comparison](results/figures/01_headline_comparison.png)

⸻

1. hidden_dag Results

mode	family	model	episodes	success	avg calls	failed progress	budget exhaustion	useful-call rate
PTC	hidden_dag	Qwen3.6-35B-A3B	40	95.0%	13.25	0.40	0.0%	80.4%
PTC	hidden_dag	GLM-5	40	97.5%	11.68	0.20	0.0%	83.9%
standard	hidden_dag	Qwen3.6-35B-A3B	40	0.0%	21.43	0.625	40.0%	59.9%
standard	hidden_dag	GLM-5	40	0.0%	32.58	0.625	85.0%	60.4%

On hidden_dag, the PTC agents did what the benchmark hoped they would do. They used the workspace as a value ledger and execution surface. They parsed the public function list, built an explicit known-value table, called functions when their inputs were available, extracted returned {"variable", "value"} pairs, and submitted the final value.

The best PTC traces read less like improvised conversation and more like a small topological executor.

The standard traces looked different. Models made real progress, but often responded to missing references by trying to refresh the world: repeating earlier function calls, reusing partial argument sets, or experimenting with empty schemas.

Some of this was model behavior. In the pre-fix run, some of it was also caused by the restore bug: the environment had forgotten values the trace had already produced.

The failure counters make the distinction concrete. For standard hidden_dag, the dominant errors were repeated calls and unknown state references. Qwen’s standard runs produced 311 repeated-call failures and 305 unknown-reference failures. GLM-5’s standard runs produced 572 repeated-call failures and 468 unknown-reference failures.

These are precisely the failures a small local known_values dictionary is supposed to prevent.

PTC agent runs was not error-free. It still produced repeated calls and some unknown references. But those errors did not usually derail the episode. The workspace made recovery cheap enough and state explicit enough that the final target value was still submitted correctly.

![hidden_dag Heatmap](results/figures/05_hidden_dag_heatmap.png)

![Call Efficiency](results/figures/04_call_efficiency.png)

⸻

1. repair_chain Results

The repair_chain result is more subtle and more interesting, but the same pre-fix caveat applies even more strongly.

mode	family	model	episodes	success	avg calls	failed progress	budget exhaustion	useful-call rate
PTC	repair_chain	Qwen3.6-35B-A3B	40	0.0%	24.80	1.00	0.0%	28.8%
PTC	repair_chain	GLM-5	27	0.0%	20.44	1.00	0.0%	26.3%
standard	repair_chain	Qwen3.6-35B-A3B	40	0.0%	41.95	0.00	45.0%	43.8%
standard	repair_chain	GLM-5	40	0.0%	53.75	0.00	77.5%	28.4%

If one reads only final reward, repair_chain looks like a failure for everyone. That is technically true under the reward function, but it is not the whole story.

PTC reached:

progress_score = 1.0

on average. That means the PTC agents generally completed the repair state transitions even though final reward stayed zero. They inspected tools, built stage maps from public inspection data, moved to the right bay, applied the stage-appropriate tool sequence, updated observation versions, and reached repaired states.

A sampled PTC trace shows the pattern plainly. The agent inspected tools, identified the stage sequence, moved to the bay, applied the full repair chain, scanned the machine at the repaired stage, and received a successful final test.

Then it called:

finish(answer="All critical machines have been repaired...")

The environment expected structured repaired object ids:

finish({"repaired": ["machine_0"]})

So the final reward stayed zero and the failure was classified as premature_submit. (This was a bug in the environemnt that has been fixed)

It says PTC succeeded at the stateful control problem. In standard mode, the pre-fix repair_chain traces were dominated by repeated calls and violated constraints. Qwen standard runs produced 871 repeated-call failures and 341 constraint violations. GLM-5 standard runs produced 880 repeated-call failures and 666 constraint violations.

Before the restore fix, these numbers should not be interpreted as pure model failure. A representative trace showed a successful move_to, followed by an apply_tool response whose current_location was null. The agent was being judged inside a world that had lost movement state between calls.

PTC repair-chain failures were different. They were dominated by repeated calls and premature submit, with much lower constraint-violation pressure. The agent was still inefficient, but it was inefficient around a working plan rather than lost before the plan.

![repair_chain Heatmap](results/figures/06_repair_chain_heatmap.png)

![Calls Distribution](results/figures/07_calls_distribution.png)

![Progress vs Success](results/figures/02_progress_vs_success.png)

⸻

1. What The Diagnostic Run Actually Shows

The trajectory audit changes the interpretation.

The cleanest claim from the PTC traces is that PTC helps when the task can be turned into a small local state machine.

On hidden_dag, the local state machine is a table of known values and a queue of callable functions. That is enough to turn a brittle multi-turn reasoning task into a deterministic execution loop. The PTC success rates in the pre-fix run were correspondingly strong: 95.0% and 97.5%.

On repair_chain, the local state machine is a repair ledger: machine, part, location, stage, observation version, stage-to-tool map, and repaired status. PTC appears to build enough of that ledger to complete the hidden state transition. The reward remains zero because of the agent, environment submission format mismatch.

Reward is the benchmark’s legal judgment. Progress is the audit trail.

In a deployment setting, the final answer-shape error would still matter. In a benchmark design setting, it tells us where to look next: not at the repair transition model, but at the finish contract.

The standard results must be rerun after the snapshot fix before making the stronger comparative claim. The pre-fix traces still revealed useful model behavior: standard agents tended to repair uncertainty by repeating tools rather than by maintaining an explicit ledger. But the old standard success rates and repair progress are not publication-grade because the harness itself was forgetting hidden mutable state.

The current conclusion is more precise:

1. PTC converts hidden DAG tasks into high-reliability executions in the current run set.
2. PTC substantially improves stateful repair progress.
3. PTC repair agents currently fail at structured final submission due to environment implementation bug, not at the underlying repair transition.
4. The initial standard results exposed a real benchmark bug: family-specific hidden state was not serialized across native tool calls.
5. The standard comparison should be rerun after the snapshot fix, and the repaired results should replace the diagnostic table.

That is a useful benchmark outcome. It exposes the promise of programmatic tool calling, the brittleness of final submission interfaces, and the value of trace-level auditing for catching benchmark infrastructure errors before they become claims.

⸻

1. What To Fix Next

Environment snapshots now include family-specific mutable state. The PTC prompt and workspace should fix the submission format.

For example, the task prompt should explicitly show the expected final shape:

finish({"repaired": ["machine_0", "machine_1"]})

for repair_chain, and:

finish({"value": 1234})

for hidden_dag.

After that, the clean rerun should answer the real question:

When the same hidden state is preserved correctly in both modes, how much does a programmable workspace improve reliability, efficiency, and recovery?

⸻

1. What Undertow Is Not

Undertow is not a claim that synthetic tasks are sufficient for agent evaluation. They are not.

It is not a claim that PTC always beats standard tool calling. It should not. 

It asks when the ability to build local machinery changes the reliability of tool use.

⸻

1. Closing

The old habit in tool-use benchmarks is to count whether the model reached the answer. Undertow keeps that count, but it is more interested in the account book: what the agent knew, what it changed, what it repeated, what it used too early, and what it failed to update.

Programmatic tool calling is only interesting if the program becomes a working instrument: a ledger, a planner, a verifier, a memory.

Undertow is a small benchmark framework for seeing whether that happens.

The first traces have spoken clearly enough for a cautious conclusion. PTC is already solving the dataflow task and largely solving the state-transition part of the repair task. 