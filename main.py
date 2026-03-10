"""
Main Execution Script — Multi-LoRA Inference Simulation.
Brings together the Global Gateway, Worker Nodes, and Trace Generation
for synchronous multi-GPU scheduling.
"""

from trace_generator import TraceGenerator
from cluster import WorkerNode, GlobalGateway

# Cluster Configuration
NUM_GPUS = 4
VRAM_PER_GPU = 3
HIDDEN_DIM, LORA_RANK = 1024, 16

def run_cluster_simulation():
    print("Initializing Multi-GPU Cluster...")
    
    # 1. Initialize Cluster Workers
    workers = []
    for i in range(NUM_GPUS):
        worker = WorkerNode(
            node_id=i, 
            vram_capacity=VRAM_PER_GPU, 
            hidden_dim=HIDDEN_DIM, 
            rank=LORA_RANK,
            prefetch=True # Logical prefetch
        )
        workers.append(worker)
        
    # 2. Initialize the Global Router (Load Balancer)
    gateway = GlobalGateway(workers, max_queue_per_worker=20)
    
    # 3. Generate a massive multi-tenant trace
    print("Generating simulated user traffic...")
    gen = TraceGenerator(seed=42)
    trace = gen.generate_skewed_trace(num_requests=200, num_adapters=15)
    
    # 4. Route all requests through the load balancer
    print("Routing requests through Affinity Gateway...")
    for req in trace:
        gateway.route(req)
        
    # 5. Execute the distributed workload
    print("Executing synchronous inference across cluster...")
    results = gateway.run_all(max_batch_size=8)
    
    # 6. Report System Health
    print("\n--- Final System State ---")
    print(f"Total Requests Processed: {len(results)}")
    total_swaps = sum(w.am.swap_count() for w in workers)
    print(f"Total Cluster VRAM Swaps: {total_swaps}")
    
    for w in workers:
        print(f"GPU {w.node_id} | VRAM: {w.am.vram_ids()} | Errors: {w.engine.stats.total_errors}")

if __name__ == "__main__":
    run_cluster_simulation()