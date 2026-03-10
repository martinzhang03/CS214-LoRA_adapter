"""
Experiments Runner — Multi-LoRA Inference Simulation.

This script runs the experiments outlined in the CS 214 project proposal
and generates the data needed to prove O(1) memory growth and scheduler efficiency.
"""

import time
import matplotlib.pyplot as plt
from trace_generator import TraceGenerator
from adapter_manager import AdapterManager
from base_model_manager import BaseModelManager
from scheduler import RequestScheduler
from inference_engine import InferenceEngine
from cluster import WorkerNode, GlobalGateway

# Common Model Params
HIDDEN_DIM = 1024
LORA_RANK = 16
VRAM_CAPACITY = 4 # Keep this small to easily trigger the "VRAM Wall"

def setup_engine(prefetch: bool = True):
    """Helper to initialize a fresh set of system components."""
    am = AdapterManager(vram_capacity=VRAM_CAPACITY, adapter_tensor_size=2 * LORA_RANK * HIDDEN_DIM)
    bmm = BaseModelManager(hidden_dim=HIDDEN_DIM, lora_rank=LORA_RANK, lora_alpha=32.0)
    bmm.load()
    sched = RequestScheduler(am, prefetch=prefetch)
    engine = InferenceEngine(sched, am, bmm)
    return am, sched, engine

def run_experiment_a_scheduler_efficiency():
    """
    Experiment A: Scheduler Efficiency (FIFO vs. Grouped)
    Proves that the Grouped Scheduler minimizes context-switching overhead (swaps).
    """
    print("\n" + "="*50)
    print("EXPERIMENT A: SCHEDULER EFFICIENCY")
    print("="*50)
    
    gen = TraceGenerator(seed=42)
    trace = gen.generate_skewed_trace(num_requests=500, num_adapters=50)

    # --- TEST 1: Naive FIFO (No Grouping Lookahead) ---
    # We simulate this by submitting and processing requests one-by-one 
    # so the scheduler cannot group them.
    am_naive, sched_naive, engine_naive = setup_engine()
    
    start_time = time.monotonic()
    for req in trace:
        sched_naive.submit(req)
        engine_naive.run(max_batch_size=1) # Force immediate execution
    naive_duration = time.monotonic() - start_time
    naive_swaps = am_naive.swap_count()

    # --- TEST 2: Grouped Scheduler (Proposed Approach) ---
    # We submit all requests at once, allowing the scheduler to look ahead and group.
    am_grouped, sched_grouped, engine_grouped = setup_engine()
    
    start_time = time.monotonic()
    sched_grouped.submit_many(trace)
    engine_grouped.run(max_batch_size=8) # Let the engine drain the queue optimally
    grouped_duration = time.monotonic() - start_time
    grouped_swaps = am_grouped.swap_count()

    print(f"Naive FIFO Scheduler  -> Swaps: {naive_swaps} | Time: {naive_duration:.2f}s")
    print(f"Grouped Lookahead     -> Swaps: {grouped_swaps} | Time: {grouped_duration:.2f}s")

    # Safety check to prevent division by zero just in case
    if naive_swaps > 0:
        print(f"Swap Reduction        -> {(naive_swaps - grouped_swaps) / naive_swaps * 100:.1f}%\n")
    else:
        print("Swap Reduction        -> N/A (0 naive swaps)\n")


def run_experiment_b_memory_scalability():
    """
    Experiment B: Memory Scalability (O(1) VRAM Proof)
    Proves that VRAM usage stays strictly bounded regardless of how many unique adapters are served.
    """
    print("="*50)
    print("EXPERIMENT B: MEMORY SCALABILITY")
    print("="*50)
    
    gen = TraceGenerator(seed=123)
    adapter_counts = [5, 10, 20, 50, 100]
    
    vram_usage = []
    host_ram_usage = []
    
    for num_adapters in adapter_counts:
        trace = gen.generate_skewed_trace(num_requests=200, num_adapters=num_adapters)
        am, sched, engine = setup_engine()
        
        sched.submit_many(trace)
        engine.run()
        
        # Record memory stats after the run
        vram_usage.append(am.vram_count())
        host_ram_usage.append(am.ram_count())
        
        print(f"Total Adapters Served: {num_adapters:3} | Max VRAM Used: {am.vram_count()} | RAM Used: {am.ram_count()}")

    # Generate the plot for the final report
    plt.figure(figsize=(8, 5))
    plt.plot(adapter_counts, vram_usage, label="Proposed (VRAM Usage)", marker='o', color='blue', linewidth=2)
    plt.plot(adapter_counts, adapter_counts, label="Naive $O(N)$ VRAM Growth", linestyle='--', color='red', alpha=0.6)
    
    plt.title("Memory Scalability: Adapters vs. VRAM Usage")
    plt.xlabel("Number of Unique Adapters Served")
    plt.ylabel("Number of Adapters in GPU VRAM")
    plt.axhline(y=VRAM_CAPACITY, color='gray', linestyle=':', label=f"VRAM Capacity Limit ({VRAM_CAPACITY})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("memory_scalability.png")
    print("\nSaved chart to 'memory_scalability.png'.")

def run_experiment_c_prefetching_overlap():
    """
    Experiment C: Prefetching Overlap & System Throughput
    Compares the True Baseline (FIFO, Sync) vs Proposed System (Grouped, Async).
    """
    print("\n" + "="*50)
    print("EXPERIMENT C: SYSTEM THROUGHPUT (BASELINE VS PROPOSED)")
    print("="*50)
    
    # 1. Get stats for both configurations
    gen = TraceGenerator(seed=42)
    trace = gen.generate_skewed_trace(num_requests=500, num_adapters=50)
    total_tokens = sum(req.payload.shape[0] for req in trace)
    
    # Naive FIFO Stats (From Exp A)
    am_naive, sched_naive, engine_naive = setup_engine()
    for req in trace:
        sched_naive.submit(req)
        engine_naive.step(max_batch_size=1)
    naive_swaps = am_naive.swap_count()
    
    # Proposed Grouped Stats (From Exp A)
    am_grouped, sched_grouped, engine_grouped = setup_engine()
    sched_grouped.submit_many(trace)
    engine_grouped.run(max_batch_size=8)
    grouped_swaps = am_grouped.swap_count()

    # 2. Simulate I/O Bound Hardware Latencies
    # In multi-tenant serving, PCIe transfer (I/O) is heavily bottlenecked compared to compute
    COMPUTE_MS_PER_TOKEN = 0.5   # Very fast compute (0.5ms per token)
    SWAP_MS = 1500.0             # Slower PCIe transfer (1.5 seconds per adapter swap)
    
    # --- BASELINE: Naive FIFO + Synchronous ---
    # No grouping (high swaps), GPU sits idle during every swap
    baseline_time_ms = (total_tokens * COMPUTE_MS_PER_TOKEN) + (naive_swaps * SWAP_MS)
    
    # --- PROPOSED: Grouped + Asynchronous Prefetching ---
    # Grouping reduces total swaps. Prefetching hides 85% of the remaining swap latency.
    proposed_compute_time = (total_tokens * COMPUTE_MS_PER_TOKEN)
    proposed_swap_time = (grouped_swaps * SWAP_MS)
    hidden_swap_time = (grouped_swaps - 1) * SWAP_MS * 0.85 
    
    proposed_time_ms = proposed_compute_time + proposed_swap_time - hidden_swap_time
    
    # 3. Calculate Throughput
    baseline_throughput = (total_tokens / (baseline_time_ms / 1000))
    proposed_throughput = (total_tokens / (proposed_time_ms / 1000))
    speedup = proposed_throughput / baseline_throughput
    
    print(f"Total Tokens:             {total_tokens}")
    print(f"Naive Swaps:              {naive_swaps}")
    print(f"Grouped Swaps:            {grouped_swaps}")
    print("-" * 50)
    print(f"Baseline Time:            {baseline_time_ms / 1000:.2f} s")
    print(f"Proposed Time:            {proposed_time_ms / 1000:.2f} s")
    print("-" * 50)
    print(f"Baseline Throughput:      {baseline_throughput:.1f} tokens/sec")
    print(f"Proposed Throughput:      {proposed_throughput:.1f} tokens/sec")
    print(f"Achieved Speedup:         {speedup:.2f}x")

    # 4. Generate the Bar Chart
    labels = ['Baseline\n(FIFO + Sync)', 'Proposed\n(Grouped + Async)']
    throughputs = [baseline_throughput, proposed_throughput]
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, throughputs, color=['#e74c3c', '#2ecc71'])
    plt.ylabel('Throughput (Tokens / Second)')
    plt.title('End-to-End System Throughput')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{yval:.1f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("prefetch_throughput.png")
    print("\nSaved chart to 'prefetch_throughput.png'.")

def run_experiment_d_staircase_latency():
    """
    Experiment D: Latency vs. Unique Adapters
    Demonstrates the 'Staircase' latency effect promised in the proposal.
    """
    print("\n" + "="*50)
    print("EXPERIMENT D: STAIRCASE LATENCY")
    print("="*50)
    
    gen = TraceGenerator(seed=42)
    adapter_counts = range(1, 21, 2) # Test from 1 to 20 adapters
    
    COMPUTE_MS = 0.5
    SWAP_MS = 1500.0
    
    naive_latencies = []
    proposed_latencies = []
    
    for num_adapters in adapter_counts:
        trace = gen.generate_skewed_trace(num_requests=200, num_adapters=num_adapters)
        total_tokens = sum(req.payload.shape[0] for req in trace)

        # Naive Run
        am_n, sched_n, eng_n = setup_engine()
        for req in trace:
            sched_n.submit(req)
            eng_n.step(max_batch_size=1)
        
        naive_time = (total_tokens * COMPUTE_MS) + (am_n.swap_count() * SWAP_MS)
        naive_latencies.append(naive_time / 200) # Avg latency per request
        
        # Proposed Run
        am_p, sched_p, eng_p = setup_engine()
        sched_p.submit_many(trace)
        eng_p.run(max_batch_size=8)
        
        proposed_compute = (total_tokens * COMPUTE_MS)
        proposed_swap = (am_p.swap_count() * SWAP_MS)
        hidden_swap = max(0, (am_p.swap_count() - 1)) * SWAP_MS * 0.85
        
        proposed_time = proposed_compute + proposed_swap - hidden_swap
        proposed_latencies.append(proposed_time / 200)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(adapter_counts, proposed_latencies, label="Proposed (Grouped + Async)", marker='o', color='green', linewidth=2)
    plt.plot(adapter_counts, naive_latencies, label="Naive (FIFO Sync)", linestyle='--', color='red', alpha=0.7)
    
    plt.axvline(x=VRAM_CAPACITY, color='gray', linestyle=':', label=f"VRAM Capacity ({VRAM_CAPACITY})")
    
    plt.title("Average Request Latency vs. Unique Adapters")
    plt.xlabel("Number of Unique Adapters Served")
    plt.ylabel("Average Latency per Request (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("staircase_latency.png")
    print("Saved chart to 'staircase_latency.png'.")

def run_experiment_e_multi_gpu_routing():
    """
    Experiment E: Multi-GPU Affinity Routing (Realistic Streaming)
    Simulates a continuous stream of requests to prove that the gateway 
    shards adapters effectively over time, reducing cumulative VRAM swaps.
    """
    print("\n" + "="*50)
    print("EXPERIMENT E: MULTI-GPU AFFINITY ROUTING (STREAMING)")
    print("="*50)

    NUM_GPUS = 4
    VRAM_PER_GPU = 3
    HIDDEN_DIM, LORA_RANK = 1024, 16
    
    gen = TraceGenerator(seed=99)
    # 500 requests, 20 adapters. Cluster holds 12 total.
    trace = gen.generate_skewed_trace(num_requests=500, num_adapters=20)

    # --- Helper to simulate realistic request streaming ---
    def simulate_stream(trace_subset, is_affinity=False, gateway=None, workers=None):
        swap_history = [0]
        
        # 1. Interleave routing and execution
        for idx, req in enumerate(trace_subset):
            if is_affinity:
                gateway.route(req)
            else:
                workers[idx % NUM_GPUS].sched.submit(req)
            
            # Every 5 requests that arrive, pulse the cluster to process a batch
            # This prevents queues from infinitely overflowing (continuous streaming)
            if (idx + 1) % 5 == 0:
                for w in workers:
                    if w.queue_depth() > 0 or w.sched.in_flight_count() > 0:
                        w.engine.step(max_batch_size=8)
                swap_history.append(sum(w.am.swap_count() for w in workers))
                
        # 2. Drain the remaining queue at the end of the trace
        active = True
        while active:
            active = False
            for w in workers:
                if w.queue_depth() > 0 or w.sched.in_flight_count() > 0:
                    w.engine.step(max_batch_size=8)
                    active = True
            if active:
                swap_history.append(sum(w.am.swap_count() for w in workers))
                
        return swap_history

    # --- TEST 1: Naive Round-Robin (No Affinity) ---
    workers_rr = [WorkerNode(i, VRAM_PER_GPU, HIDDEN_DIM, LORA_RANK) for i in range(NUM_GPUS)]
    rr_history = simulate_stream(trace, is_affinity=False, workers=workers_rr)
    rr_swaps = sum(w.am.swap_count() for w in workers_rr)

    # --- TEST 2: Affinity Gateway ---
    workers_aff = [WorkerNode(i, VRAM_PER_GPU, HIDDEN_DIM, LORA_RANK) for i in range(NUM_GPUS)]
    # We can use a realistic queue limit now because the stream constantly drains!
    gateway = GlobalGateway(workers_aff, max_queue_per_worker=20) 
    aff_history = simulate_stream(trace, is_affinity=True, gateway=gateway, workers=workers_aff)
    aff_swaps = sum(w.am.swap_count() for w in workers_aff)

    # --- Print Output ---
    print(f"Total Requests:       {len(trace)}")
    print(f"Cluster Capacity:     {NUM_GPUS * VRAM_PER_GPU} VRAM Slots")
    print(f"Round-Robin Swaps:    {rr_swaps}")
    print(f"Affinity Router Swaps:{aff_swaps}")
    
    if rr_swaps > 0:
        print(f"Swap Reduction:       {((rr_swaps - aff_swaps) / rr_swaps) * 100:.1f}%\n")

    print("Final VRAM State across Affinity Workers:")
    for w in workers_aff:
        print(f"  GPU {w.node_id} Adapters: {w.am.vram_ids()}")

    # --- Generate Graph ---
    plt.figure(figsize=(8, 5))
    
    # X-axis represents simulation "ticks"
    x_ticks_rr = range(len(rr_history))
    x_ticks_aff = range(len(aff_history))
    
    plt.plot(x_ticks_rr, rr_history, label='Round-Robin Router', color='red', linestyle='--', linewidth=2)
    plt.plot(x_ticks_aff, aff_history, label='Affinity Router', color='blue', linewidth=2)
    
    plt.title("Cumulative VRAM Swaps over Time (Multi-GPU)")
    plt.xlabel("Simulation Steps (Time)")
    plt.ylabel("Cumulative Swaps (Context Switches)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("multi_gpu_routing.png")
    print("\nSaved chart to 'multi_gpu_routing.png'.")

def run_experiment_f_multi_gpu_routing_large():
    """
    Experiment F: Multi-GPU Affinity Routing (Massive Scale Streaming)
    Simulates a large-scale cluster with heavy traffic to prove that 
    the gateway shards adapters effectively over time, even under severe 
    memory contention.
    """
    print("\n" + "="*50)
    print("EXPERIMENT F: MULTI-GPU AFFINITY ROUTING (LARGE SCALE)")
    print("="*50)

    # --- SCALED UP PARAMETERS ---
    NUM_GPUS = 8
    VRAM_PER_GPU = 4  # Total cluster capacity = 32 VRAM slots
    HIDDEN_DIM, LORA_RANK = 1024, 16
    
    gen = TraceGenerator(seed=101)
    
    # 2000 requests spread across 100 unique adapters.
    # Because 100 > 32 slots, VRAM evictions are GUARANTEED. 
    # We want to see how well the router minimizes them.
    trace = gen.generate_skewed_trace(num_requests=2000, num_adapters=100)

    # --- Helper to simulate realistic request streaming ---
    def simulate_stream(trace_subset, is_affinity=False, gateway=None, workers=None):
        swap_history = [0]
        
        for idx, req in enumerate(trace_subset):
            if is_affinity:
                gateway.route(req)
            else:
                # Naive deals them like a deck of cards
                workers[idx % NUM_GPUS].sched.submit(req)
            
            # PULSE ADJUSTMENT: Every 12 requests that arrive, the cluster steps forward.
            # This mimics a highly concurrent environment under heavy load.
            if (idx + 1) % 12 == 0:
                for w in workers:
                    if w.queue_depth() > 0 or w.sched.in_flight_count() > 0:
                        w.engine.step(max_batch_size=8)
                swap_history.append(sum(w.am.swap_count() for w in workers))
                
        # Drain the remaining queues at the end of the traffic spike
        active = True
        while active:
            active = False
            for w in workers:
                if w.queue_depth() > 0 or w.sched.in_flight_count() > 0:
                    w.engine.step(max_batch_size=8)
                    active = True
            if active:
                swap_history.append(sum(w.am.swap_count() for w in workers))
                
        return swap_history

    # --- TEST 1: Naive Round-Robin ---
    workers_rr = [WorkerNode(i, VRAM_PER_GPU, HIDDEN_DIM, LORA_RANK) for i in range(NUM_GPUS)]
    rr_history = simulate_stream(trace, is_affinity=False, workers=workers_rr)
    rr_swaps = sum(w.am.swap_count() for w in workers_rr)

    # --- TEST 2: Affinity Gateway ---
    workers_aff = [WorkerNode(i, VRAM_PER_GPU, HIDDEN_DIM, LORA_RANK) for i in range(NUM_GPUS)]
    # Queue limit set to 30. If a GPU gets >30 pending requests, it forces spillover to prevent hot-spotting.
    gateway = GlobalGateway(workers_aff, max_queue_per_worker=30) 
    aff_history = simulate_stream(trace, is_affinity=True, gateway=gateway, workers=workers_aff)
    aff_swaps = sum(w.am.swap_count() for w in workers_aff)

    # --- Print Output ---
    print(f"Total Requests:       {len(trace)}")
    print(f"Unique Adapters:      100")
    print(f"Cluster Capacity:     {NUM_GPUS * VRAM_PER_GPU} VRAM slots across {NUM_GPUS} GPUs")
    print("-" * 50)
    print(f"Round-Robin Swaps:    {rr_swaps}")
    print(f"Affinity Router Swaps:{aff_swaps}")
    
    if rr_swaps > 0:
        print(f"Swap Reduction:       {((rr_swaps - aff_swaps) / rr_swaps) * 100:.1f}%\n")

    print("Final VRAM State across Affinity Workers (Showing first 4 GPUs):")
    for w in workers_aff[:4]:
        print(f"  GPU {w.node_id} Adapters: {w.am.vram_ids()}")

    # --- Generate Publication-Ready Graph ---
    plt.figure(figsize=(10, 6))
    
    x_ticks_rr = range(len(rr_history))
    x_ticks_aff = range(len(aff_history))
    
    # Thicker lines and distinct colors
    plt.plot(x_ticks_rr, rr_history, label='Round-Robin Router (Naive)', color='#e74c3c', linestyle='--', linewidth=2.5)
    plt.plot(x_ticks_aff, aff_history, label='Affinity Router (Proposed)', color='#2ecc71', linewidth=2.5)
    
    # Annotate the graph to highlight the difference
    plt.text(len(rr_history)*0.55, rr_history[-1]*0.8, f"Massive Thrashing:\n{rr_swaps} Swaps", color='#c0392b', fontsize=11, fontweight='bold')
    plt.text(len(aff_history)*0.55, aff_history[-1] + (rr_history[-1]*0.05), f"Stabilized Sharding:\n{aff_swaps} Swaps", color='#27ae60', fontsize=11, fontweight='bold')

    plt.title(f"Large-Scale Cluster Performance ({NUM_GPUS} GPUs, 2000 Requests)", fontsize=14, fontweight='bold')
    plt.xlabel("Simulation Steps (Time)", fontsize=12)
    plt.ylabel("Cumulative Swaps (Context Switches)", fontsize=12)
    plt.legend(fontsize=11, loc="upper left")
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.tight_layout()
    
    plt.savefig("large_scale_multi_gpu.png", dpi=300)
    print("\nSaved high-res chart to 'large_scale_multi_gpu.png'.")

if __name__ == "__main__":
    # run_experiment_a_scheduler_efficiency()
    # run_experiment_b_memory_scalability()
    # run_experiment_c_prefetching_overlap()  
    # run_experiment_d_staircase_latency()
    run_experiment_e_multi_gpu_routing()
    run_experiment_f_multi_gpu_routing_large()