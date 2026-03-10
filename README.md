# Scalable Multi-LoRA Inference: Optimizing Multi-Tenant LLM Serving

**CS 214 — Jack He, Allen Wang, Jinyuan Zhang, Francis Wang**

This project explores how to serve many LoRA adapters (specialist models) on a single GPU without hitting the "VRAM wall" or slowing to a crawl, and scales this approach to a **multi-GPU cluster** using memory-aware request routing. We use a simulation to benchmark our approach: reproducible traces, a two-tier memory manager, a grouped scheduler, and a simulated execution engine.

---

## What We're Trying to Achieve

### The Problem: The "VRAM Wall"

- **Trend:** Moving from one generalist model (e.g. Llama-3) to many specialists (Legal-Llama, Medical-Llama, Code-LoRA, etc.). We want to run multiple specialized adapters on one machine.
- **Tech:** LoRA (Low-Rank Adaptation) stores only the *delta* (ΔW), typically <1% of base model size.
- **Bottleneck:** The base model is still large (7B+ parameters). Loading one full instance per adapter gives **O(N) VRAM** and is infeasible.
- **Conflict:** Switching adapters often causes context-switching overhead; keeping all adapters in VRAM is impossible.
- **Goal:** Serve 100+ adapters on a single GPU with bounded memory and good throughput/latency.

### Gaps in Existing Work

- **vLLM / PagedAttention:** Addresses KV-cache, not multi-adapter weight management.
- **S-LoRA / Punica:** Batch multiple LoRA weights in one kernel, but still assume synchronous swapping—the GPU sits idle while weights are reloaded, and there is no prefetching from the request queue.

Our approach combines:
1. **LRU paging** — Move adapter weights between Host RAM and GPU VRAM (fixed-size “shelf”).
2. **Request grouping** — Reorder the queue so all requests for the same adapter are processed together, reducing swap frequency.
3. **Asynchronous prefetching** — Load the next adapter into a buffer while the current one is still running.
4. **Affinity-Based Routing (Multi-GPU)** — A global gateway that distributes requests across a cluster based on memory locality, naturally sharding adapters to specific nodes to prevent cross-node cache thrashing.

---

## How We Achieve It: Simulation Design


We do **not** run real LLM inference. Instead we simulate the **lifecycle of a request** and the **management of adapter weights** so we can measure throughput, latency, and VRAM usage in a reproducible way. The simulation has five main blocks.

### 1. Trace Generator (Workload)
**Purpose:** Produce a reproducible stream of “incoming requests” to benchmark the system.
**Design:** A script that outputs a list of request objects. Each object has:
- `request_id`
- `adapter_id` (e.g. `"Medical"`, `"Legal"`)
- `burst_size` (simulated token count)
**Logic:** Generate **clusters** of the same `adapter_id` to mimic real batching opportunities, plus **random outliers** to stress LRU eviction.

### 2. Adapter Manager (VRAM / RAM Control)

**Purpose:** Implement the “VRAM shelf” with **LRU (Least Recently Used)** eviction.
**Design:** A Python class that maintains two tiers:
- **Host RAM:** A large dictionary of CPU tensors (effectively unbounded).
- **GPU VRAM:** A **fixed-size pool** of GPU tensors (e.g. max K adapters).
**Logic:**
- On adapter request: if it is already in the GPU pool, mark it as recently used and return.
- If not in GPU: **evict** the least recently used adapter from VRAM to Host RAM, then **load** the requested adapter from Host RAM (or create it if new) into VRAM.
- VRAM pool size is strictly enforced → **O(1) memory growth** with respect to number of adapters.
**Implementation:** `adapter_manager.py` — `AdapterManager` class.

### 3. Grouped Scheduler (Concurrency)
**Purpose:** Cut down context-switching overhead by reordering the queue.
**Design:** Middleware between the Trace Generator and the Execution Engine.
**Logic:** Use a **look-ahead window** (e.g. next 20 requests). Reorder so that all requests sharing an `adapter_id` are processed in one contiguous batch before the Adapter Manager performs a weight swap. This reduces how often we evict and reload.

### 4. Execution Engine (Simulated Compute)
**Purpose:** Simulate the time spent on “real” GPU compute.
**Design:** A loop that runs a dummy PyTorch op (e.g. large matrix multiply) for a duration proportional to the request’s token count (`burst_size`).
**Logic:** Use `torch.cuda.Stream` so that an **asynchronous prefetcher** can load the next adapter’s weights into a buffer while the current batch is running—**hiding** load latency behind compute.

### 5. Multi-GPU Cluster & Global Gateway (Distributed Routing)

**Purpose:** Scale the inference pipeline horizontally across multiple simulated GPUs.
**Logic:** Wraps the single-GPU pipeline into isolated `WorkerNode`s. A `GlobalGateway` acts as a load balancer, routing requests to the GPU that already holds the required adapter in its queue or VRAM (Affinity). It intentionally breaks affinity only if a GPU's queue depth exceeds a threshold, preventing hot-spotting via dynamic replication.

---

## Test Data and Experiments

### Synthetic Traces

Two trace types are used:

| Trace | Requests | Unique adapters | Purpose |
| :--- | :--- | :--- | :--- |
| **Uniform** | 100 | 5 | Baseline “standard serving”; even distribution. |
| **Skewed** | 500 | 50 | Exceeds VRAM capacity; stresses LRU eviction. |

### Required Experiments

| Test case | Description | Expected metric |
| :--- | :--- | :--- |
| **Baseline (naive)** | No grouping, no prefetching; reload weights for every request. | Low throughput (~15–50 tokens/sec). |
| **Memory scalability** | Increase unique adapters from 1 to 50. | Naive: VRAM grows **O(N)**. Ours: VRAM **flat**, O(1). |
| **Scheduler efficiency** | Compare “random order” vs “grouped order” processing. | Grouped order: **staircase** latency—overhead mainly at adapter swaps. |
| **Prefetching overlap** | Total time with vs without `cudaMemcpyAsync` (async prefetch). | Total time **decreases** as weight loading is hidden behind compute. |
| **Multi-GPU Routing** | Stream 500 requests (Uniform & Skewed) across a 4-GPU cluster. | Affinity Gateway shows up to **~81% swap reduction** over Naive Round-Robin. |
| **Large-Scale Routing**| Stress test 2000 requests, 100 adapters (Uniform & Skewed) on 8 GPUs. | Proves cluster stability under extreme memory contention (up to **~85% swap reduction**). |

Testing is done in **separate** test/experiment scripts and notebooks, not inside the core implementation modules.

---

## Expected Results (From the Simulation)

If the design is implemented and tuned correctly, the simulation should show:

1. **Throughput:** About **3×–5×** improvement over naive reload-per-request, by avoiding long “GPU idle” periods.
2. **Latency:** A plot of **average latency vs number of unique adapters** should look like a **staircase** (steps at swap points) rather than an exponential blow-up.
3. **VRAM stability:** Once the adapter pool exceeds the VRAM limit, **VRAM usage** should flatten to a **horizontal line**, demonstrating that the paging system keeps memory bounded.
4. **Massive Distributed Cache Hits (Actual):** Our `GlobalGateway` successfully auto-shards adapters across multiple GPUs under both uniform and skewed workloads. 
    * Under continuous streaming (4 GPUs, 20 adapters), the affinity logic reduced cache swaps by **48.0%** for uniform traffic and **81.6%** for skewed traffic with hot-spots. 
    * Under extreme scaling (8 GPUs and 100 adapters), it reduced swaps by **38.1%** for uniform traffic and a massive **85.8%** for skewed traffic, dropping context switches from 1,324 down to just 188. This proves the gateway effectively handles viral traffic spikes.

---

## Repository Structure

- **`adapter_manager.py`** — Adapter Manager: two-tier RAM/VRAM pool with LRU eviction (implementation only; no test harness inside this file).
- **Trace generator, Grouped Scheduler, Execution Engine** — To be added as separate modules.
  - **`base_model_manager.py`** — Owns the frozen base weights in VRAM.
  - **`scheduler.py`** — Request grouping and look-ahead prefetching.
  - **`inference_engine.py`** — Pipeline coordinator.
  - **`cluster.py`** — `WorkerNode` wrappers and `GlobalGateway` load balancer.
  - **`trace_generator.py`** — Workload creation.
- **Tests and experiments** — Implemented in separate test/experiment code as per the project outline.
  - **`main.py`** — Synchronous multi-GPU execution script.
  - **`experiments.py`** — Complete test suite generating `matplotlib` graphs.

---

## Dependencies

- **Python 3.x**
- **PyTorch** (for `AdapterManager` tensors and, later, the execution engine and prefetch simulation)
- **Matplotlib** (for generating experiment evaluation charts)

Install with: `pip install torch matplotlib` (or your project’s full `requirements.txt` when added).

---

## Summary

This project **intends** to show that multi-tenant LoRA serving can be made scalable on a single GPU by:
- Capping VRAM with an **LRU adapter manager** (O(1) VRAM growth),
- **Grouping requests** by adapter to reduce swaps, and
- **Prefetching** the next adapter while the current one is in use.
- **Affinity Routing** across multiple GPUs to shard the workload and eliminate cross-node thrashing.

We **achieve** the evaluation through a **simulation** (trace generator → adapter manager → grouped scheduler → execution engine) and separate test/experiment code, leading to measurable throughput, latency, and VRAM stability results for the final report.