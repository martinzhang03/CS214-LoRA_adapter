"""
Trace Generator — Workload creation for Multi-LoRA Inference Simulation.

Purpose: Generate reproducible streams of InferenceRequest objects to simulate
different real-world multi-tenant serving scenarios.
"""

import random
import torch
from typing import List
from scheduler import InferenceRequest

class TraceGenerator:
    def __init__(self, seed: int = 42):
        """
        Initialize the generator with a seed for reproducible traces.
        """
        self.seed = seed
        random.seed(self.seed)

    def generate_uniform_trace(
        self, 
        num_requests: int = 100, 
        num_adapters: int = 5,
        min_burst: int = 10,
        max_burst: int = 150
    ) -> List[InferenceRequest]:
        """
        Baseline Trace: Evenly distributes requests across a small set of adapters.
        Purpose: Tests standard serving without hitting the VRAM wall.
        """
        random.seed(self.seed) # Reset seed per trace generation for consistency
        adapters = [f"adapter_{i}" for i in range(num_adapters)]
        trace = []

        for i in range(num_requests):
            adapter_id = random.choice(adapters)
            burst_size = random.randint(min_burst, max_burst)
            
            dummy_tensor = torch.randn(burst_size, 1024)
            
            req = InferenceRequest(
                request_id=f"uni_{i}",
                adapter_id=adapter_id,
                payload=dummy_tensor
            )
            trace.append(req)

        return trace

    def generate_skewed_trace(
        self, 
        num_requests: int = 500, 
        num_adapters: int = 50,
        cluster_probability: float = 0.8,
        max_cluster_size: int = 15,
        min_burst: int = 10,
        max_burst: int = 150
    ) -> List[InferenceRequest]:
        """
        Stress Trace: Exceeds VRAM capacity. Creates clusters of the same adapter
        to simulate batching opportunities, mixed with random outliers to trigger
        LRU evictions.
        """
        random.seed(self.seed)
        adapters = [f"adapter_{i}" for i in range(num_adapters)]
        trace = []
        i = 0

        while i < num_requests:
            # Decide whether to create a cluster of the same adapter, or a random outlier
            if random.random() < cluster_probability:
                # --- CLUSTER ---
                adapter_id = random.choice(adapters)
                cluster_size = random.randint(5, max_cluster_size)
                # Ensure we don't exceed the total requested number of traces
                cluster_size = min(cluster_size, num_requests - i) 
                
                for _ in range(cluster_size):
                    burst_size = random.randint(min_burst, max_burst)
                    dummy_tensor = torch.randn(burst_size, 1024)
                    trace.append(InferenceRequest(
                        request_id=f"skew_{i}",
                        adapter_id=adapter_id,
                        payload=dummy_tensor
                    ))
                    i += 1
            else:
                # --- OUTLIER ---
                adapter_id = random.choice(adapters)
                burst_size = random.randint(min_burst, max_burst)
                dummy_tensor = torch.randn(burst_size, 1024)
                trace.append(InferenceRequest(
                    request_id=f"skew_{i}",
                    adapter_id=adapter_id,
                    payload=dummy_tensor
                ))
                i += 1

        return trace

# ==========================================
# Quick Verification / Example Usage
# ==========================================
if __name__ == "__main__":
    generator = TraceGenerator(seed=123)
    
    print("Generating Uniform Trace...")
    uniform_trace = generator.generate_uniform_trace(num_requests=5, num_adapters=2)
    for req in uniform_trace:
        print(f"  {req.request_id} -> {req.adapter_id} (Tokens: {req.payload['burst_size']})")
        
    print("\nGenerating Skewed Trace (showing first 15)...")
    skewed_trace = generator.generate_skewed_trace(num_requests=50, num_adapters=20)
    for req in skewed_trace[:15]:
         print(f"  {req.request_id} -> {req.adapter_id} (Tokens: {req.payload['burst_size']})")