# vllm-configurator

A web app that generates a `vllm serve` configuration for your deployment: GPU, model, vLLM version, workload profile
and latency targets. It also generates the configuration of the llm-d smart routing (EPP) that spreads the requests
over those vLLM instances. It explains every value it picks and flags what is left to verify.

The app's interface, justifications and warnings are in French. This README quotes the interface labels as they
appear on screen.

- [What the app does](#what-the-app-does)
- [Getting started](#getting-started)
- [Maintenance: architecture and catalog](#maintenance-architecture-and-catalog)

---

## What the app does

### The problem

The vLLM parameters that weigh most on performance (`--max-num-batched-tokens`, `--max-num-seqs`,
`--long-prefill-token-threshold`, speculative decoding, CUDA graph capture, `--gpu-memory-utilization`) depend on each
other, on the workload profile and on the vLLM version. Their sizing rules exist, but they are scattered across PRs,
release notes and recipes, and they change from one version to the next. The app applies these rules reproducibly
and shows its reasoning.

### Inputs

| Group | What you provide |
|---|---|
| Hardware | GPU type, GPUs per instance (tensor parallel), number of instances |
| Model and version | Catalog model, vLLM version |
| Workload profile | Usage type (chat, agentic, RAG, extraction, summary, scoring, batch), interactive or batch mode, ISL and OSL at P50 / P95 / P99, expected prefix cache hit rate, share of cold requests (prefix missing from the cache), target requests in flight, tokens to prefill for a warm request, wait between two requests of a session; or import a [working-set](https://github.com/T0mSIlver/working-set) trace, which fills these fields |
| SLOs | Target TTFT p95 and ITL p95 (optional) |
| Production measurements (optional) | ITL (or TPOT) p95 and TTFT p95 measured with the generated config: they take precedence over the estimate; the generation latency recalibrates the decode part of the model, the TTFT its prefill part. Or import a `ws replay` result |
| Speculative decoding | On/off, `k`, draft sampling method, measured acceptance length, tolerance to a lossy verification (FLy) |
| Context and extra arguments (optional) | Target `max-model-len` (with YaRN context extension when the model has one), extra vLLM flags and environment variables |
| Startup logs (optional) | `GPU KV cache size`, block size and `gpu-memory-utilization` read in the logs of one instance |
| llm-d routing (optional) | llm-d version, approximate or precise prefix index, prefix shared by all sessions, forced weights, values read in the EPP log and hit rate measured per pod |

Only the ISL P50 and the OSL P50 are required. A missing input is replaced by a displayed assumption, or the app asks
for it explicitly when it cannot do without it.

### Outputs

- **The configuration**, in two formats: the `args:` block to paste into a KServe or Kubernetes manifest, and the
  equivalent `vllm serve` command.
- **A justification for every value**: the rule applied, the formula and the worked computation, the vLLM default,
  the sources, and a ✅ verified or 🔶 to-verify status.
- **Warnings** ranked by severity: model not supported by the version, lossy FLy in agentic use, suspicious OSL,
  limiting KV, unconfirmed data, breaking changes of the version.
- **Computed indicators**: actual ISL_eff/OSL ratio compared to the ratio needed to fill the budget, contexts the KV
  can hold at P50/P95/P99, tokens consumed per decode step, tokens per MoE expert, sequence cap imposed by vLLM with
  DSpark.
- **A step-by-step test plan**: for each `max-num-seqs` value to validate, the flags to set, the load to inject, the
  expected ranges of ITL, TPOT, TTFT and KV usage, and the number of instances needed if the step holds. With agentic
  sessions, the sessions to replay per step and the `ws replay` command. Measuring the steps settles the sizing; the
  estimate is there to choose where to test. An indicative estimate at the generated config's operating point
  (bottleneck, fixes) remains available. See [Test steps and SLOs](#test-steps-and-slos).
- **The capacity in agentic sessions** per instance, for the generated config and for the best vLLM setting found.
  See [Capacity in sessions](#capacity-in-sessions).
- **The assumptions**: every value the config rests on, with its origin (entered, trace, replay, logs, or assumed by
  the app), shown first at step ② so that you can correct them.
- **Verification**: a command to check that every flag exists on the deployed image, and the list of Prometheus
  metrics to watch.
- **The llm-d configuration**: the `EndpointPickerConfig` (prefix index and scorer weights) derived from the vLLM
  config, with its justifications and the EPP log lines to check after deployment. See
  [llm-d routing](#llm-d-routing).
- **A review report** (Markdown and JSON) to send to a reviewer, a person or an agent: the questions to answer, every
  assumption with its origin, the config and its justifications, what the app expects, the measurements and the
  known limits of the tool. See [Review report](#review-report).
- **A YAML scenario file**, exportable from and re-importable into the interface: inputs, SLOs and measurements, plus
  a snapshot of the generated config. See [Save and reload a scenario](#save-and-reload-a-scenario).

### What you can rely on

- **No unknown flag.** Every flag and every JSON key emitted was extracted from the vLLM source code at the tag of the
  chosen version. This matters from 0.33 on, whose Rust frontend refuses to start on an unknown flag.
- **Rules that follow the version.** For instance, with DSpark at k = 3, each decoding sequence consumes 6 budget
  tokens in 0.32 and 4 in 0.33. FLy is only offered from 0.33 on.
- **No invented specification.** Model cards are pinned to an exact Hugging Face revision. When the KV size per token
  cannot be computed reliably, the app asks for the value from the logs instead of estimating it.

### Limits

- The app measures nothing: it applies rules. The step latencies are **ranges** (×0.8 to ×2.5 without measurement)
  from a roofline model whose efficiencies are assumed (🔶). Measuring the steps decides; entering the measured
  generation latency and TTFT recalibrates each its part of the model (±15 %).
- The TTFT p95 mostly depends on the shape of the load: share of cold requests, actual hit rate, bursts. A load test
  with random prompts, or with a fixed OSL started all at once, does not reproduce production (see the step protocol).
- Without startup logs, the KV size is a rough estimate (upper bound): rerun the computation with the real value after
  the first deployment.
- The llm-d scorer weights are choices of the app (🔶), set after the configuration proposed by the knowledge base:
  the hit rate measured per pod is what validates them. The precise mode only generates the known part of its
  configuration (block size, `--kv-events-config`); the indexer and the tokenizer sidecar are to be set up following
  the llm-d guide.

### Current catalog

| Type | Content |
|---|---|
| GPU | NVIDIA B300 |
| Models | DeepSeek V4.1 Flash (build 1017), Qwen3.8 Flash Next, GLM-5.3 Flash, Mistral Large 4 (dedicated image required) |
| vLLM versions | 0.31.0, 0.32.0, 0.33.0 |
| llm-d (EPP) | RHOAI 3.5.x |

### Example: the reference case

DeepSeek V4.1 Flash on 2 pods of 4 × B300 (TP4), vLLM 0.33.0, agentic chat, ISL P50 24K, KV of 86 M tokens per pod
read in the logs. Generated configuration (the app writes its YAML comments in French; translated here):

```yaml
args:
  # --- Parallelism ---
  - --tensor-parallel-size
  - "4"
  - --data-parallel-size
  - "1"

  # --- Scheduler ---
  - --max-num-seqs
  - "512"
  - --max-num-batched-tokens
  - "8192"
  - --long-prefill-token-threshold
  - "2048"
  - --long-prefill-token-threshold-adaptive
  - --enable-chunked-prefill

  # --- KV cache / prefix caching ---
  - --enable-prefix-caching
  - --kv-cache-dtype
  - "fp8"
  - --gpu-memory-utilization
  - "0.92"
  # no block-size: forced to 128 by the backend, do not force it (§5)
  # no kv-offloading-size: 3,592 P50 contexts fit in the KV, offload not needed
  # no max-model-len: default = maximum context of the model (1,048,576)

  # --- Speculative decoding ---
  - --speculative-config
  - '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","enable_adaptive_verification":false}'

  # --- CUDA graphs ---
  - --compilation-config
  - '{"max_cudagraph_capture_size":2048}'

  # --- Frontend ---
  - --api-server-count
  - "2"
  # no renderer-num-workers: the vLLM default is enough.
```

For this case, the app also reports that the OSL P50 of 4,000 tokens is about 16 times the published agentic median
(247), and that the ISL P99 of 666K needs qualifying: measured per call, or summed per session?

---

## Getting started

### 1. Prerequisites

- Python 3.11 or later.
- [uv](https://docs.astral.sh/uv/) for the environment and the dependencies.
- No GPU needed: the app does not run vLLM.

### 2. Installation

From the project root:

```bash
uv sync
```

uv creates a `.venv` environment and installs the project and its dependencies in it (Streamlit, pydantic, PyYAML,
pytest).

### 3. Start the interface

```bash
uv run streamlit run src/vllm_configurator/web/app.py
```

The app opens on <http://localhost:8501>. It is a five-step workflow; the step bar at the top lets you go back and
forth, and every value you entered is kept from one step to the next.

| Step | What you do | What the app does |
|---|---|---|
| **① Votre besoin** | Hardware, model, vLLM version, workload (or a working-set trace), SLOs. « Affiner » holds the optional inputs. | Nothing yet: only ISL P50 and OSL P50 are required to move on. |
| **② Hypothèses et config** | Read what the app assumed and correct it in ① if it does not match your case. | Lists every assumption with its origin (entered, trace, replay, logs, or assumed), then the proposed config, why each value, what it expects (checks, capacity in sessions) and the llm-d config. |
| **③ Tester** | Deploy the config and run the step plan. | Gives the steps, the load to inject (requests in flight, or sessions and the `ws replay` command), the protocol, the pass criteria and the flag check command. |
| **④ Mesurer et ajuster** | Import the `ws replay --out` result, or type the measurements (startup logs, Prometheus latencies, acceptance length). | Recalibrates the model at the measured load, compares the estimated and measured capacity, and proposes the setting that holds more sessions. |
| **⑤ Exporter** | Download the review report and the scenario. | Writes the report for a reviewer, a person or an agent (see [Review report](#review-report)). |

### 4. First try: the reference case

Click **« Charger le cas de référence (§1) »** (load the reference case) at step ①. The form fills with the real
deployment described in `docs/vllm-config-knowledge-base.md`, and the app jumps to step ② with the assumptions and the
configuration.

### 5. Describe your own deployment (step ①)

The results are recomputed on every change.

| Section | Advice |
|---|---|
| **Matériel et modèle** | « GPU par instance » is the tensor parallel of one pod; « Instances » is the number of independent pods. Pick the vLLM version actually deployed: several rules depend on it. |
| **Charge** | Measure ISL and OSL **per model call**, not per conversation turn or per session. Target requests in flight are a total over all instances: running + queued requests, **not users** (an agentic user only has a request running part of the time). Measurement: `vllm:num_requests_running` + `vllm:num_requests_waiting` summed over the pods. For agentic sessions, import a working-set trace instead (see [Agentic sessions](#agentic-sessions-working-set-replay)): it fills the lengths, the cache, the cold requests and the waits. |
| **Objectifs (SLO)** | Optional. They do not change the generated config; they set the pass criteria of the steps. Choose what the generation target is about: **ITL** (`vllm:inter_token_latency_seconds`, interval between two outputs, i.e. one step) or **TPOT** (`vllm:request_time_per_output_token_seconds`, time per token). With speculative decoding the ITL rises a little but the TPOT is divided by ~a: for the user experience, the TPOT is what counts. |
| **Affiner** | Hit rate (empty: the value published for your usage type), share of cold requests (empty: 5 %, capped to stay consistent with the hit rate; they make the TTFT p95), warm-request tokens, wait between requests, speculative decoding (leave `k` empty for the model's recommended value; only allow a lossy verification for an A/B test: FLy can change the outputs, which is risky for tool calls), context and extra arguments (below), CPU limit, llm-d routing (see [llm-d routing](#llm-d-routing)). |

Measurements (startup logs, latencies, acceptance length, llm-d logs) are entered at step ④.

#### Extend the context and add vLLM arguments

**Target `max-model-len`.** Empty, the model serves its native context and `--max-model-len` is not emitted. Below
the native context, the value is emitted as is. Beyond it, the app applies the context extension of the model card
when it has one. Example with Qwen3.8 Flash Next (native 262,144, 1,000,000 via YaRN according to the model card):

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3.8-Flash-Next \
  --max-model-len 1000000 \
  --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"original_max_position_embeddings":262144,"factor":4.0}}}'
```

- `factor` is the smallest multiple of 0.5 that covers `max-model-len / native context`: 2.0 for 524,288 and 4.0 for
  1,000,000, as in the model card.
- The `rope_parameters` block is passed whole: vLLM replaces it as a block, whereas it merges `text_config`.
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` is required from 0.31 to 0.33: for YaRN, vLLM does not apply the factor to the
  maximum derived from `config.json` and otherwise refuses to start. Step ② then shows an `env:` block
  to set on the container, and the `vllm serve` command is prefixed with it.
- YaRN is static: it applies to every request and can degrade short texts. Only extend the context if the traffic
  needs it.

**Extra vLLM arguments.** A YAML mapping, one entry per flag. A `dict` is emitted as JSON (so is a JSON string),
`true` emits the bare flag, `--no-<flag>: true` disables a boolean.

```yaml
--enable-log-requests: true
--served-model-name: qwen-long
--hf-overrides: {text_config: {rope_parameters: {rope_type: yarn, factor: 2.0}}}
```

- Each flag is checked against `catalog/vllm/extracted/<version>.yaml`. An unknown flag is dropped, with a critical
  warning: the Rust frontend of vLLM ≥ 0.33 would refuse to start.
- A flag the app already generates is replaced by the value given. For `--max-num-seqs`, `--max-num-batched-tokens`,
  `--speculative-config`, `--tensor-parallel-size` and `--max-model-len`, a warning reminds you that the analysis is
  still computed on the generated value.
- A hand-written `--hf-overrides` replaces the catalog's extension. With a `max-model-len` beyond the native context,
  the app still sets `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.

**Environment variables.** A YAML mapping (`VLLM_LOGGING_LEVEL: DEBUG`), added to the `env:` block.

The context, the arguments and the variables are saved in the scenario files.

#### Save and reload a scenario

Step ⑤ exports the whole form (hardware, model, workload profile, SLOs, measurements, startup logs) to a YAML file;
« 💾 Reprendre un scénario enregistré » at step ① re-imports it to fill the form. Name the scenario before exporting:
the name is used for the file name.

```yaml
kind: vllm-configurator/scenario
version: 1
name: Prod agentique B300
saved_at: '2026-11-12T10:00:00'
request:                  # the only part read back on import
  gpu_id: b300
  gpus_per_instance: 4
  num_instances: 2
  model_id: deepseek-v4.1-flash-1017
  vllm_version: 0.33.0
  workload: {usage_type: agentic, mode: interactive, isl: {p50: 24000}, osl: {p50: 250}, target_concurrency: 800}
  slo: {ttft_p95_ms: 2000.0, itl_p95_ms: 40.0}
generated:                # informational snapshot (args, SLO verdict), ignored and recomputed on import
  args: [...]
```

The files can be versioned next to the manifests. The import also accepts a bare request, in the format of
`tests/golden/reference_case/input.yaml`. It rejects a file whose GPU, model or vLLM version is missing from the
catalog, and says why.

### 6. Deploy and test (step ③)

1. **Check the flags on the image actually deployed**, with the command under « Vérifier les flags sur l'image » of
   step ③. It lists the flags the image knows; every flag of the config must show up:
   ```bash
   vllm serve --help=all | grep -oE -- "--(no-)?(max-num-seqs|max-num-batched-tokens|...)\b" | sort -u
   ```
2. **Paste the `args:` block** of step ② into the vLLM container definition of your manifest. This block does not
   contain the model name, which KServe usually provides elsewhere. The `vllm serve` command does include it.
3. **Deploy**, then check in the logs that vLLM starts without errors and without OOM during the CUDA graph capture.
4. **Run the step plan** of step ③, with `ws replay` for agentic sessions.

### 7. Measure and adjust (step ④)

The KV size depends on `gpu-memory-utilization`, `max-num-seqs` and the CUDA graph capture: it changes with the new
configuration.

1. Enter the new `GPU KV cache size` line and the block size of the startup logs: the app recomputes `max-num-seqs`
   and the indicators.
2. Import the `ws replay --out` result, or enter the latencies measured at the generated config's step: the model is
   recalibrated and the ranges of the other steps narrow.
3. Measure the acceptance length `a = 1 + spec_decode_num_accepted_tokens / spec_decode_num_drafts` and enter it: it
   replaces the a = 3 assumption.
4. Read the capacity in sessions and the flags of the best setting; deploy it and replay again to validate it.

### 8. Export for review (step ⑤)

#### Review report

The report gathers what a reviewer needs to judge whether the configuration fits the workload
(`src/vllm_configurator/report.py`):
- the review questions and the expected answer format (overall verdict, blocking points, points to check, proposed
  corrections);
- every assumption with its origin, so the reviewer can challenge the ones the app assumed;
- the SLOs, the generated config and the justification of every parameter, the llm-d config;
- what the app expects: checks, capacity in sessions, bottlenecks, fixes, and the values the latency model ran on;
- the test plan, the measurements (startup logs, latencies, working-set replay steps);
- the known limits of the tool.

It comes as Markdown (`<name>-rapport.md`, to send to an agent or a person) and as JSON (`<name>-rapport.json`, for a
tool). Step ⑤ also exports the scenario (`.yaml`), to reload at step ①.

### llm-d routing

With several independent instances, no cache is shared: a session routed to another pod than the one holding its
prefix redoes its whole prefill. The llm-d EPP picks the pod of each request by combining weighted scorers. The app
sets this `EndpointPickerConfig` from the vLLM config: it reads the engine's facts (`DeploymentFacts`: KV per
instance, block size, `max-num-seqs`, number of instances) without recomputing them. The code is in
`src/vllm_configurator/llmd/`, the catalog in `catalog/llmd/<version>.yaml`.

**The prefix index (approximate mode).** The EPP keeps, per pod, an LRU of the block hashes of the prompts it routed.

| Parameter | Rule |
|---|---|
| `blockSizeTokens` | vLLM KV block size (one index block = one vLLM block), at least 64: below that, the EPP overrides it |
| `lruCapacityPerServer` | KV per instance / `blockSizeTokens`, to track a pod's whole cache. Too small: false misses, random routing |
| `autoTune` | `false`: the capacity is fixed, the automatic correction is not confirmed |
| `maxPrefixTokensToMatch` | At least the ISL P95 and the shared prefix, rounded up to a power of 2, at least the default (131,072); `maxPrefixBlocksToMatch` is the same in blocks |

**The profile weights.**

| Scorer | Rule |
|---|---|
| `prefix-cache-scorer` | From the prefill saved by affinity: (1 − h without affinity) / (1 − h), with h without affinity = h shared + (h − h shared) / N. Gain ≥ 3 → 5, ≥ 1.5 → 3, else 2 |
| `kv-cache-utilization-scorer` | From the KV usage at the generated config's step: ≥ 80 % → 3, ≥ 50 % → 2, else 1 |
| `queue-scorer` | 2, or 3 if the requests in flight per pod reach 80 % of `max-num-seqs` |
| `no-hit-lru-scorer` | 2: spreads the new sessions, whose prefix is known nowhere |

The shared prefix (system prompt, tools) is cached on every pod: declaring it lowers the expected gain of affinity.
Left empty, the app assumes 0, the case where affinity matters most.

**Force a value.** Like the extra vLLM arguments, the « Paramètres llm-d forcés (YAML) » field replaces a computed
value of the prefix plugin (one line per parameter, e.g. `lruCapacityPerServer: 700000`). A name unknown to the
catalog or a value of the wrong type is ignored, with a critical warning. The app also reports what the forced value
breaks: index too small for the KV, block below the minimum, window shorter than the shared prefix, precise block size
different from vLLM's. The weights are forced in the « Poids … » fields of « Affiner » (step ①). At step ②, the
« 📖 Comprendre les paramètres llm-d » panel explains each parameter and each scorer, with its default and the
proposed value.

For the reference case, the app finds the configuration of §9.4 of the knowledge base: block 128,
`lruCapacityPerServer` 673,639, weights 2 / 1 / 5 / 2 (queue / kv / prefix / no-hit). Without affinity, the 95.7 %
hit rate would drop to ~48 % and the uncached prefill would be multiplied by 12.

**Precise mode.** The pods publish their KV events over ZMQ: the app generates `--kv-events-config` (to add to the
vLLM args, with the `POD_IP` variable on the container) and `tokenProcessorConfig.blockSize` = vLLM block size, which
it therefore requires. It reports when this mode is of little use (lightly used KV, rare evictions) or, conversely,
when the approximate index is likely to drift (KV more than 80 % used).

**Close the loop after deployment.** Step ② gives the lines expected in the EPP log (`Prefix DataProducer
initialized`, `EPP config after phase two`). Enter at step ④ (llm-d fields) what the log actually shows and the hit
rate of the worst pod: the app computes the share of the cache the deployed index forgets and explains a hit rate that is
too low. The EPP config is recomputed on every change of the vLLM config, since the KV size depends on it.

### Test steps and SLOs

An unmeasured latency model can be off by a factor of 2 or 3. The app therefore promises no verdict: it gives a
**step plan** that brackets the limit, and the measurement does the sizing. The code is in
`src/vllm_configurator/analysis.py`.

**The steps.** They are `max-num-seqs` values (16, 32, 64… 2048), capped by what the KV holds and by the target
requests in flight per instance. For each of them, the app gives:

- the flags to set: `--max-num-seqs`, the CUDA graph capture and, if the prefill load requires it, a higher
  `--max-num-batched-tokens` budget;
- the load to inject, as many requests in flight as sequences;
- the expected ranges of ITL, TPOT, TTFT, throughput and KV usage;
- a forecast: ✅ if even the pessimistic case meets the SLOs, 🛑 if even the optimistic case misses them, ⚠️ in
  between;
- the number of instances needed if this step is the last one to pass: ⌈target requests in flight / max-num-seqs⌉.

The app keeps the steps around the expected limit: two safe steps, the uncertain zone and the first step likely to
fail. The generated config's step is marked ★.

**Protocol.** Each step is validated on one instance, in a closed loop, by replaying real traces (same ISL, OSL and
thinking as in production). The steady state must last at least three times the duration of a request. A step passes
if the generation latency and the TTFT p95 meet the target, with no queue (`vllm:num_requests_waiting` ≈ 0), no
preemption (`vllm:num_preemptions`) and `vllm:kv_cache_usage_perc` below 90 %. The first failing step sets the
capacity of an instance at the previous step.

#### Agentic sessions: working-set replay

When the wait between two requests of a session is known (entered, or read off an imported trace), each step also
gives the **sessions to replay**: requests in flight × (request duration + wait) / duration, for the longest then the
shortest estimated duration. It also gives the KV those sessions hold: a session waiting on a tool keeps its context
in cache, and beyond the pool vLLM evicts the oldest ones, whose next turn redoes its whole prefill. A step whose
sessions exceed 90 % of the KV goes to ⚠️. The protocol then gives the `ws replay` command of the step, with
thresholds derived from the SLOs.

**Import a working-set trace** (`src/vllm_configurator/traces.py`, « 📼 Importer une trace » box of the workload
profile). A `ws trace pi|claude` trace (format `workingset-agent-trace` v1 or v2) only holds sizes and waits. The app
reads it with the same accounting as `ws replay`, under a perfect cache:

| Request | Tokens to prefill |
|---|---|
| First turn of a session, or a prompt that shrinks (compaction, branch): cold | prompt − prefix shared by all sessions (llm-d section) |
| Next turn: warm | prompt − previous prompt − previous reply (without the reply if the chat template drops it from the history: untick the box) |

From it, the app derives:
- the ISL and the OSL at P50, P95 and P99;
- the token-weighted hit rate, and the share of cold requests;
- the P50 and P95 of the tokens of a warm request, which set the dispersion σ instead of the heuristic;
- the mean wait between two requests. Like `ws replay`, the app replaces the inactive waits (error, resume, session
  left open) by the mean human wait.

With a number of sessions and the service times of a v2 trace, it also splits these sessions into requests in flight
and paused sessions. The hit rate and the cold share of a perfect cache are floors: evictions and misrouting only add
cold requests. Compare them to the measured hit rate and to the eviction share `ws replay` reports.

Three points to respect so that the replay looks like production:
- **`--shared-prefix-tokens`.** By default, `ws replay` takes the smallest first prompt of the trace as the prefix
  shared by all sessions. With sessions that start past 100k tokens, those 100k tokens become shared and the first
  turns are nearly warm. Behind a router (llm-d), it also flattens the prefix scores between pods and weakens session
  affinity. The protocol therefore sets this flag to the declared shared prefix.
- **`--api chat`.** It goes through the chat template and the renderer; the completions API bypasses them.
- **The spec decoding acceptance.** The replayed text is synthetic and its output has a forced length: enter the
  acceptance measured during the test in the app, so that the right factor is recalibrated.

working-set judges the p90 TTFT of warm requests and the p10 decode speed. The thresholds derived from the SLOs
(`--max-warm-ttft-s`, `--min-decode-tok-s`) are therefore looser than p95 SLOs: the verdict of a step is still read on
the server metrics.

#### From the replay to the setting

The intended workflow:
1. describe the workload, the hardware and the model, then import an agentic trace;
2. deploy the generated config and replay it with `ws replay … --out run.json`;
3. import `run.json` at step ④ (« Importer un rejeu working-set »);
4. read the capacity in sessions and the setting that gives more, at step ④;
5. export the review report at step ⑤.

On import of the result (`traces.profile_run`), the app reads every judged step again with working-set's window rule:
- the requests in flight (mean);
- the TTFT p95, cold requests included, and the p90 TTFT of warm requests;
- the TPOT p95, i.e. (last token − first) / (tokens − 1) for each request;
- the hit rate and the share of cold requests, from the `cached_tokens` vLLM returns, which requires
  `--enable-prompt-tokens-details`;
- the evictions, and the server metrics if the replay read them.

One step is used for the recalibration: by default the largest passing step, the closest to the limit. Its latencies
recalibrate the model **at the measured load**, brought down to one instance (one per replay URL, else the number of
« Instances »), not at the target concurrency. Its measured hit rate and cold share replace the trace's, evictions and
routing included.

working-set only measures the TPOT, not the ITL. With an ITL target, the decode part is recalibrated on the TPOT, and
the ITL p95, which the steps loaded with prefill make, stays extrapolated. Read it on
`vllm:inter_token_latency_seconds` during the replay to recalibrate it.

#### Capacity in sessions

`analysis.session_capacity`: for each setting, the app looks for the largest number of requests in flight per
instance that passes every check at the pessimistic end of the estimate. It turns that into sessions, then caps them by
the KV of the sessions: active and waiting, shared prefix counted once, 90 % of the pool. It compares the generated
config to the best setting of `--max-num-seqs`, `--max-num-batched-tokens`, `--long-prefill-token-threshold` and the
frontend. It gives the flags to change, the limit reached (ITL, TPOT, TTFT, prefill load, KV of the sessions,
frontend) and the levers beyond the setting: offloading the KV of waiting sessions, `gpu-memory-utilization`, TP,
instances, P/D disaggregation. If the estimated capacity differs from the one validated by working-set by more than
×1.4, the app says so: recalibrate before following the proposed setting, then validate that setting with a new
replay.

#### The estimation model

It is used to place the steps:

| Quantity | Computation |
|---|---|
| Lengths | ISL and OSL means derived from the percentiles (lognormal fitted on P50/P95, corrected if the P99 reveals a heavier tail), not from the P50 |
| KV usage | S × (mean ISL + mean OSL / 2) / KV per instance, KV brought back to the config's `gpu-memory-utilization` if the logs were read at another value |
| Tokens to prefill | A distribution, not a mean: a share of cold requests prefilled in full (`cold_request_fraction`), and for the others an uncached delta independent of the history, heavy-tailed (`uncached_tokens_sigma`, or the warm-request percentiles of a trace), fitted to match the hit rate |
| Prefill load | uncached prefill requested per step (S·a/OSL × E[tokens to prefill]) / (B − S·d), in steady state (§3.1) |
| Step | sum of the operators (`src/vllm_configurator/perf.py`), each bounded by its roofline max(memory reads, compute): dense weights, MoE experts hit, attention (KV reads; FLOPs over the context, including the indexer of a sparse attention), TP all-reduce over NVLink, fixed overheads (higher on a mixed step) and per sequence, drafter pass |
| ITL / TPOT | ITL p95 = step loaded with prefill; TPOT p95 = mean step / a of the 5 % least accepted requests (dispersion `acceptance_dispersion`) |
| TTFT | 95th percentile of the sum frontend (HTTP, EPP, tokenization) + wait for a slot + wait for the prefill (M/G/c queue, prompt variance included) + prefill of the prompt chunked by the long prefill threshold, computed over the distribution of the tokens to prefill |

Without measurement, durations are given from ×0.8 to ×2.5; recalibrated on a measured latency, at ±15 %
(`perf_uncertainty`), separately for the decode part (generation latency) and the prefill part (TTFT). The
efficiencies and costs (`perf_hbm_efficiency`, `perf_mfu`, `perf_overhead_per_layer_ms`, `perf_nvlink_efficiency`,
`perf_drafter_step_fraction`, `perf_sparse_indexer_cost_ratio`, `frontend_python`…) and the thresholds are heuristics
of `catalog/workload_profiles.yaml`, all 🔶. The attention cost is derived from config.json (heads, dimensions, layers
with non-linear attention, top-k of a sparse attention); the `performance` block of a model card can set it
(`attention_flops_per_context_token`, `attention_sparse_flops_per_token`, `attention_sparse_tokens`). Without a
`performance` block in the model card, nor `bandwidth` and `compute` in the GPU card, the steps have no latency
ranges: only the capacity is checked.

**The indicative estimate**, folded under the plan, keeps the analysis at the generated config's operating point. It
gives the bottlenecks (prefill, KV, sequence slots, decode, prefill interference, ITL floor) and fixes: a setting at
constant capacity, instances, wider TP, `gpu-memory-utilization`, DP-attention, prefix cache, P/D disaggregation, PCP.
A setting is only proposed if it holds even in the pessimistic case; it still has to be validated by the steps.

### Frontend: `--api-server-count` and `--renderer-num-workers`

The GPU is not the only possible bottleneck. Before and after it, the **Python frontend** of `vllm serve` (the
default; the Rust frontend, `VLLM_USE_RUST_FRONTEND=1`, is opt-in and not modelled) does all the CPU work of a
request. What follows was checked in the source code of vLLM 0.31.0, 0.32.0 and 0.33.0 (behaviors
`frontend.api_servers` and `frontend.renderer_pool` of each version's catalog).

**What each flag does.**

| Flag | What it changes | What it does not change |
|---|---|---|
| `--api-server-count` (default: `data_parallel_size`, so 1 in DP1) | N API processes on a shared port. Each has its asyncio loop, its GIL, its renderer and its detokenization. | The engine core stays single: beyond that, it or the GPU is the limit. |
| `--renderer-num-workers` (default: 1) | Size of the thread pool of each API process, which runs the chat template (HF and DeepSeek V4 renderers) and the tokenization of the prompt. | Neither the outputs (detokenization, streaming, parsers, on the asyncio loop), nor multimodal (a dedicated thread is enforced), nor the offline mode (`LLM`). |

The chat template is Python (Jinja, or `encode_messages` for DeepSeek V4 and V4.1): it holds the GIL even on a pool
thread, so more threads do not parallelize it. Tokenization does benefit. The verified path is `encode` →
transformers `_encode_plus` → `tokenizers` `encode_batch`, whose Rust core runs outside the GIL (`py.allow_threads`,
renamed `py.detach`). It was read in transformers 5.10.3, 5.16.1 and 5.20.2 and in `tokenizers` 0.21.1, 0.23.2 and
0.23.3, the versions pinned by vLLM 0.31 to 0.33. Only extracting the text and converting the result to Python lists
keep the GIL: the app counts 10 % for them (`tokenize_gil_share`, 🔶). The `fastokens` backend
(`VLLM_USE_FASTOKENS=1`) was not reviewed.

**The model.** Per API process, the app estimates:
- the GIL load: requests/s × (loop + message JSON + chat template) + streamed outputs/s × cost of one output;
- the pool load: requests/s × (chat template + tokenization), spread over the threads.

The traffic comes from the GPU step: S·a/OSL requests per step, and one output per sequence and per step. The waits on
the GIL (M/G/1) add to the ITL; those of the pool (M/G/c) add to the TTFT. Rule R-FE-01/02 sizes to the smallest
number that keeps every GIL and every pool below 70 %. It uses the fast steps of the estimate (more traffic, so the
cautious case for the frontend) and keeps the vLLM defaults when they hold. The unit costs are heuristics
(`frontend_python`, 🔶).

**When it is worth touching them.**

| Scenario | What saturates | Lever |
|---|---|---|
| Long outputs, large MoE, 30 to 50 ms steps, spec decoding (the reference case at moderate concurrency) | Nothing: a few requests/s, GIL at ~40 % | Keep the defaults |
| Many parallel streams and short steps: fast model, high `max-num-seqs`, no spec decoding (one output per token) | The GIL, at ~90 % from the outputs | `--api-server-count`; the renderer cannot help |
| Short calls on long contexts: agentic at an OSL of ~250, RAG, classification, scoring | The pool: tens of requests/s × 20–30 ms of rendering each | `--renderer-num-workers` for the tokenization, `--api-server-count` if the chat template dominates |
| Multimodal | The dedicated multimodal thread | `--api-server-count` (one multimodal thread per process) |
| Full instances (no target concurrency) | Often the outputs: 512 streams × ~25 steps/s ≈ 13,000 outputs/s | `--api-server-count 2` (deviation documented in the golden test) |

Speculative decoding lightens the frontend: one output per step carries ~a tokens.

**What it costs, and when it becomes the bottleneck.**
- **CPU**: ~1 core per saturated API process, plus the tokenization threads, on top of the engine core (~1 core) and
  of one worker per GPU (~1 each). A CPU limit set too low throttles (CFS) the GPU workers, which is worse than a slow
  frontend. Fill in « Cœurs CPU par instance »: the app caps the API processes and shows the need (indicator
  `frontend_cpu`).
- **Constraints**: with N > 1, Prometheus switches to multiprocess mode (check that the EPP and the dashboards read the
  aggregated metrics). N > 1 is also incompatible with `VLLM_ALLOW_RUNTIME_LORA_UPDATING`. Elastic EP brings it back
  to 1, and each process loads its tokenizer in host memory.
- **Signs of saturation**: an API process at ~100 % of a core (`top -H`, `pidstat`, `py-spy dump`). Another sign: a
  client-side TTFT or ITL that drifts further and further from the server's as the load rises.
- **Beyond**: the engine core (serialization of inputs and outputs) becomes the next bottleneck. More API processes
  change nothing there: you need more instances, or DP inside the instance.

**Rust frontend.** vLLM ignores `--api-server-count` (multithreaded frontend, no GIL) and refuses unknown flags. Its
cost is not modelled: compare it in an A/B test at the same step.

### 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'vllm_configurator'` | `uv sync --reinstall-package vllm-configurator` |
| « Taille du KV inconnue » (unknown KV size) | The model has an architecture whose KV cannot be computed reliably: enter the `GPU KV cache size` read in the logs. |
| Critical warning « absente du registre de vLLM » | The chosen version does not support this model. Use the dedicated image given, then check the flags on it. |
| Model, GPU or version missing from the lists | Add it to the catalog (see below). |
| Unreadable accented characters in a Windows terminal | Prefix the scripts with `PYTHONUTF8=1`. |

### 10. Run the tests

```bash
uv run pytest
```

The tests cover the formulas (with the numbers of the knowledge base), the rules and their dependency on the version,
the consistency of the catalog, the golden test, the trace and replay imports, and the interface.

---

## Maintenance: architecture and catalog

The original knowledge base is `docs/vllm-config-knowledge-base.md`. The checks made in the vLLM source code,
including several corrections to that base, are in `docs/verification-vllm-0.33.0.md`.

### Architecture

```
catalog/                    Versioned data (YAML), separate from the logic
  gpus/<id>.yaml            GPUs: HBM, compute capability family…
  models/<id>.yaml          Models: pinned config.json, KV, block size, native spec decoding, vLLM support
  vllm/<version>.yaml       vLLM versions: defaults, behaviors, emitted flags, breaking changes
  vllm/extracted/*.yaml     Flags and JSON keys extracted from the source code at the tag (generated)
  llmd/<version>.yaml       llm-d (EPP): plugins, defaults read in the logs, behaviors, weight heuristics
  workload_profiles.yaml    Workload profiles, heuristics, metrics
src/vllm_configurator/
  catalog/                  Pydantic schemas and loading + cross-validation
  formulas.py               Formulas of §3, pure functions
  perf.py                   Model of one scheduler step (sum of per-operator rooflines) and queues
  frontend.py               CPU model of the Python frontend (GIL, renderer pool)
  analysis.py               Test steps, SLO checks, capacity in sessions, bottlenecks and fixes
  scenario.py               YAML scenario files (form export / import)
  traces.py                 working-set traces (ws trace) and replay results (ws replay --out)
  rules/                    Rules: one pure function (RuleContext, state) -> RuleOutput per decision
  engine.py                 Runs the rule pipeline, produces a GenerationResult
  render.py                 `args` YAML (KServe/K8s), command line, verification command
  llmd/                     llm-d target: EndpointPickerConfig rules and YAML rendering
  targets/                  Target registry: `vllm` and `llmd_epp`
  web/app.py                Streamlit interface
tests/
  unit/                     Formulas, rules, catalog, traces, UI
  golden/reference_case/    Reference case: §1 inputs, §7 expected config, justified deviations
tools/extract_vllm_flags.py Extraction of the flags of a vLLM version from GitHub
```

Principles:

- **Rules hard-code no version.** They read the behaviors of the version's catalog
  (`ctx.behavior("spec_decode.extra_draft_slots")`, etc.). Changing versions means changing files.
- **No unknown flag is emitted.** Rendering refuses a flag missing from `catalog/vllm/extracted/<version>.yaml`. The
  engine also refuses an unknown `--speculative-config` key.
- **Honesty about uncertainty.** Every `unverified` input used by a rule produces a « À vérifier » warning. A missing
  input produces either a displayed assumption or a blocking error that asks for the value.
- **Always emit.** The flags the config relies on are emitted even when they equal the default. The exceptions are
  `--block-size` (never force a size the backend imposes), `--kv-offloading-size` (its mere presence turns offload on)
  and `--max-model-len` (default = maximum context of the model). Their reason is written as a comment in the YAML.
- **Decoupled targets.** `GenerationResult.facts` (`DeploymentFacts`) exposes the KV per instance and its source, the
  block size, `max_num_seqs`, etc. The `llmd_epp` target consumes these facts
  (`lruCapacityPerServer = formulas.lru_capacity_per_server(kv, block)`) without touching the vLLM engine.

### Add an llm-d version

Copy `catalog/llmd/rhoai-3.5.yaml` to `catalog/llmd/<id>.yaml` (the file name must equal `id`). Read the defaults in
the EPP startup log (`Prefix DataProducer initialized`, `EPP config after phase two`) and set them to
`status: verified` with that log as the source. Keep as `unverified` what was not read (location of the parameters,
effect of autoTune, precise-mode parameters).

### Add a GPU

The Claude Code skill `vllm-configurator-catalog` (`.claude/skills/vllm-configurator-catalog/`) automates this
procedure and the model one. By hand:

1. Create `catalog/gpus/<id>.yaml` (the file name must equal `id`), after `b300.yaml`.
2. `memory.hbm_gb` in decimal GB, from the vendor datasheet. If only a system page gives a total, divide it, but set
   `status: unverified` and explain the cross-check in `note`.
3. `cudagraph_family`: `blackwell_dc` for a compute capability 10.x, else `other`. It is the only information vLLM
   uses for the CUDA graph capture cap.
4. `bandwidth.hbm_tb_s` and `compute.dense_tflops` (fp4, fp8, bf16, **dense** values per GPU): optional, they feed
   the latency model of the test steps.
5. Validate (see below).

The HBM is only used to estimate the KV when the startup logs are not provided.

### Add a model

1. Identify the exact Hugging Face repository and collect the facts at a pinned revision:
   ```bash
   PYTHONUTF8=1 uv run python .claude/skills/vllm-configurator-catalog/scripts/hf_model_facts.py <org/repo>
   ```
2. Read the recipe `https://recipes.vllm.ai/<org/repo>`: spec decoding method and `k`, KV dtype, minimum version,
   dedicated image.
3. Create `catalog/models/<id>.yaml` after the existing cards. Rules:
   - `hf_revision`: full 40-character SHA; the `source` URLs point to that revision.
   - `kv_cache.bytes_per_token`: leave `null` if the architecture is exotic (linear or hybrid attention, compression,
     shared KV, indexers). The app then requires the « GPU KV cache size » line of the logs, which is intended. A
     standard formula (GQA, plain MLA) goes into `method`, with `status: unverified`.
   - `block_size.forced_by_backend`: only from a real log or an explicit instruction of the recipe.
   - `speculative.native_method` must be listed in `speculative_methods` of the `extracted` file of the target version.
   - `context_extension` (optional): only from the model card or the recipe. `native_max_model_len`, reachable
     `max_model_len`, `override_path` (e.g. `[text_config, rope_parameters]`) and the complete `rope_parameters`
     block **without** `factor`, which the app computes.
   - `performance` (optional, for the latency ranges of the steps): parameters activated per token from the model card
     or the recipe, size of the routed experts from `config.json`, bytes per parameter according to the quantization.
4. Validate.

The field-by-field details are in `.claude/skills/vllm-configurator-catalog/references/fields.md`.

### Add a vLLM version

1. **Extract the flags** at the version's tag:
   ```bash
   uv run python tools/extract_vllm_flags.py v0.34.0
   ```
   The tool generates `catalog/vllm/extracted/0.34.0.yaml`: CLI flags, `--speculative-config` and
   `--compilation-config` keys, spec decoding methods, registry architectures. The source files are downloaded into
   `.cache/vllm-src/` and read as text, never executed.
2. **Copy the previous catalog** (`catalog/vllm/0.33.0.yaml` → `0.34.0.yaml`), then update `version`, `source_tag`
   and `verified_on`.
3. **Check every behavior again in the source code at the new tag** and fix `value`, `source` and `status`. Where to
   look:

   | Behavior | File, symbol |
   |---|---|
   | `defaults.*` | `vllm/config/cache.py`, `scheduler.py`, `speculative.py` (field values) |
   | `long_prefill.*` | `vllm/v1/core/sched/scheduler.py`, `schedule()` (`num_eligible_reqs`, `adaptive_long_prefill_threshold`) |
   | `spec_decode.verification_tokens_in_budget` | `scheduler.py`, computation of `num_new_tokens` for running requests |
   | `spec_decode.extra_draft_slots` | `vllm/config/speculative.py`, `max_num_new_slots_for_drafting` (table in the docstring) |
   | `spec_decode.dspark_active_seq_cap` | `vllm/config/vllm.py`, `max_draft_seqs` |
   | `spec_decode.fly`, `adaptive_verification`, `draft_lm_head_quantization` | `vllm/config/speculative.py` |
   | `cudagraph.*` | `vllm/config/vllm.py`, `_set_cudagraph_sizes`; `vllm/config/compilation.py` |
   | `frontend.*` | `vllm/entrypoints/cli/serve.py` (`run_multi_api_server`), `vllm/renderers/base.py`, `hf.py`, `deepseek_v4.py`, `vllm/v1/engine/async_llm.py` (`output_handler`) |

   What could not be read in the code stays `status: unverified`, with the source used (release notes…). The app will
   show it as « à vérifier ».
4. **Update** `breaking_changes` and `notes` from the release notes.
5. If a rule needs a new behavior, add the key to **every** version catalog, with each one's own value, then read it
   in the rule through `ctx.behavior(...)`. `tests/unit/test_catalog.py` checks that every version defines the
   behaviors in use.
6. Validate.

### Validate the catalog

```bash
uv run python -c "from vllm_configurator.catalog import load_catalog; c = load_catalog(); print(c.validate() or 'OK')"
uv run pytest
```

### Golden test

`tests/golden/reference_case/` holds the §1 inputs (`input.yaml`), the §7 config copied as is (`expected.yaml`) and
the accepted deviations (`deviations.yaml`). The test fails if the generated config differs from §7 without the
deviation being listed and justified, or if a listed deviation no longer happens. Current deviations:

- flags added by the « always emit » policy (`--tensor-parallel-size 4`, `--data-parallel-size 1`,
  `--enable-chunked-prefill`, keys `rejection_sample_method` and `enable_adaptive_verification`);
- `--long-prefill-token-threshold-adaptive`, without which the dynamic behavior of §2.5 is not active in 0.33;
- same 8192 budget, but a corrected justification: with 1 + k verification tokens per sequence, the ratio needed is
  4.0, and 8192 is kept for the ITL in interactive mode;
- `--gpu-memory-utilization` left at the vLLM default (0.92) instead of 0.8: vLLM profiles the CUDA graph memory, so
  0.8 would count it twice;
- `--api-server-count 2`: at full load, the streamed outputs would saturate the GIL of a single API process.
