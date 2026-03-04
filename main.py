from adapter_manager import AdapterManager        # Part 2
from scheduler import RequestScheduler, InferenceRequest  # Part 1
from base_model_manager import BaseModelManager   # Part 4
from inference_engine import InferenceEngine      # Part 3

HIDDEN, RANK = 1024, 16
am  = AdapterManager(vram_capacity=4, adapter_tensor_size=2*RANK*HIDDEN)
bmm = BaseModelManager(hidden_dim=HIDDEN, lora_rank=RANK, lora_alpha=32.0)
bmm.load()

sched  = RequestScheduler(am, prefetch=True)
engine = InferenceEngine(sched, am, bmm)

sched.submit(InferenceRequest("r1", adapter_id="lora-A"))
sched.submit(InferenceRequest("r2", adapter_id="lora-B"))
sched.submit(InferenceRequest("r3", adapter_id="lora-C"))
sched.submit(InferenceRequest("r4", adapter_id="lora-D"))
sched.submit(InferenceRequest("r5", adapter_id="lora-E"))
sched.submit(InferenceRequest("r6", adapter_id="lora-F"))
sched.submit(InferenceRequest("r7", adapter_id="lora-G"))
sched.submit(InferenceRequest("r8", adapter_id="lora-H"))
sched.submit(InferenceRequest("r9", adapter_id="lora-I"))
sched.submit(InferenceRequest("r10", adapter_id="lora-J"))

results = engine.run()   # drains the queue