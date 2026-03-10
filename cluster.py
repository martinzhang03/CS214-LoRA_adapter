"""
Multi-GPU Cluster Simulation — Part 5 of the Multi-LoRA Inference Simulation.

Purpose: Simulate a distributed routing layer that scales the inference pipeline
horizontally. It balances data locality (affinity) with queue depth (load balancing)
across multiple independent simulated GPU workers.

Responsibilities
----------------
- Encapsulate the single-node components (AdapterManager, RequestScheduler,
  BaseModelManager, InferenceEngine) into isolated WorkerNode instances representing
  individual GPUs.
- Expose a GlobalGateway to act as the single entry point/load balancer for all 
  incoming InferenceRequest objects from the trace generator.
- Route incoming requests to specific WorkerNodes based on an affinity algorithm:
  prefer GPUs that already have the target adapter in VRAM, while avoiding GPUs
  that have exceeded a maximum queue depth threshold (preventing hot-spots).
- Orchestrate cluster-wide execution via a run_all() loop that pulses each
  worker's engine to simulate concurrent, multi-node inference.

Design notes
------------
- Composition over modification: The cluster logic strictly wraps the existing 
  single-GPU classes. The core modules (Parts 1-4) remain completely unaware 
  that they are running inside a distributed system.
- Affinity vs. Load Balancing: The routing decision is a strict hierarchy. It
  prioritizes VRAM cache hits (zero swap cost) but will intentionally break
  affinity if a worker's queue becomes too deep, trading a swap penalty to 
  prevent severe queueing latency.
- Concurrency simulation: GlobalGateway.run_all() uses a round-robin loop, 
  stepping each worker's engine one batch at a time to mimic parallel execution 
  across the physical cluster.
"""

from typing import List, Optional
from adapter_manager import AdapterManager
from base_model_manager import BaseModelManager
from scheduler import RequestScheduler, InferenceRequest
from inference_engine import InferenceEngine, InferenceResult

class WorkerNode:
    """Encapsulates a single simulated GPU stack."""
    
    def __init__(self, node_id: int, vram_capacity: int, hidden_dim: int, rank: int, prefetch: bool = True):
        self.node_id = node_id
        
        # In a real system, device would be f"cuda:{node_id}". 
        # For simulation, we can just use the default logic or CPU.
        self.am = AdapterManager(vram_capacity=vram_capacity, adapter_tensor_size=2 * rank * hidden_dim)
        self.bmm = BaseModelManager(hidden_dim=hidden_dim, lora_rank=rank, lora_alpha=32.0)
        self.bmm.load()
        
        self.sched = RequestScheduler(self.am, prefetch=prefetch)
        self.engine = InferenceEngine(self.sched, self.am, self.bmm)

    def queue_depth(self) -> int:
        return self.sched.queue_depth()

    def has_adapter(self, adapter_id: str) -> bool:
        # Check physical VRAM first
        if self.am.is_in_vram(adapter_id):
            return True
        # If not in VRAM, check if it is already scheduled to be loaded!
        if adapter_id in self.sched.pending_adapters():
            return True
            
        return False


class GlobalGateway:
    """Routes requests to WorkerNodes using an Affinity + Load Balancing algorithm."""
    
    def __init__(self, workers: List[WorkerNode], max_queue_per_worker: int = 50):
        self.workers = workers
        self.max_queue_per_worker = max_queue_per_worker

    def route(self, request: InferenceRequest) -> None:
        """
        Routing logic:
        1. Find workers that already have the adapter in VRAM (Affinity).
        2. Filter out workers that are overloaded (Hot-spot prevention).
        3. If a valid affinity worker exists, route to it.
        4. If not (cache miss or overloaded), route to the worker with the absolute lowest queue.
        """
        # 1. Affinity Check
        affinity_workers = [w for w in self.workers if w.has_adapter(request.adapter_id)]
        
        # 2. Prevent Hot-spotting (filter overloaded workers)
        valid_affinity = [w for w in affinity_workers if w.queue_depth() < self.max_queue_per_worker]
        
        # 3. Decision
        if valid_affinity:
            # Route to the affinity worker with the shortest queue
            target = min(valid_affinity, key=lambda w: w.queue_depth())
        else:
            # Cache miss OR all affinity workers are overloaded.
            # Fallback: Route to the worker with the absolute shortest queue.
            target = min(self.workers, key=lambda w: w.queue_depth())
            
        target.sched.submit(request)

    def run_all(self, max_batch_size: int = 8) -> List[InferenceResult]:
        """Drains all worker queues and aggregates the results."""
        all_results = []
        # Simulate round-robin concurrent execution across the cluster
        active = True
        while active:
            active = False
            for worker in self.workers:
                if worker.sched.queue_depth() > 0 or worker.sched.in_flight_count() > 0:
                    results = worker.engine.step(max_batch_size=max_batch_size)
                    all_results.extend(results)
                    active = True
        return all_results