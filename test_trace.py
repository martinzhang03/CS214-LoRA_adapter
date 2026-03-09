from trace_generator import TraceGenerator

def test_clustering():
    gen = TraceGenerator(seed=42)
    trace = gen.generate_skewed_trace(num_requests=500, num_adapters=50)
    
    # Count how many times we switch adapters
    swaps = 0
    for i in range(1, len(trace)):
        if trace[i].adapter_id != trace[i-1].adapter_id:
            swaps += 1
            
    print(f"Total Requests: {len(trace)}")
    print(f"Total Adapter Swaps (Naive FIFO): {swaps}")
    print(f"Average Cluster Size: {len(trace) / swaps:.2f}")

if __name__ == "__main__":
    test_clustering()