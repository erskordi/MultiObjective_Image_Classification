import torch
from torch_flops import TorchFLOPsByFX
import torchvision.models as models
from zeus.monitor import ZeusMonitor

def calculate_flops_and_memory(model=None, input_tensor=None, device="cuda"):
    if model is None or input_tensor is None:
        raise ValueError("Both model and input_tensor must be provided.")
    
    # Before an operation
    mem_before = torch.cuda.memory_allocated() if device == "cuda" else torch.memory_allocated()

    # 1. Run the model once for accurate measurement (optional, for time/memory)
    with torch.no_grad():
        model(input_tensor)

    # After the operation
    mem_after = torch.cuda.memory_allocated() if device == "cuda" else torch.memory_allocated()

    # 2. Initialize the FLOPs counter
    flops_counter = TorchFLOPsByFX(model)

    # 3. Propagate the input
    flops_counter.propagate(input_tensor)

    # 4. Print results
    flops_counter.print_result_table() # Detailed per-layer table
    total_flops = flops_counter.print_total_flops(show=True)

    
    print(f"Memory allocated before: {mem_before / 1024**2:.2f} MB")
    print(f"Memory allocated after: {mem_after / 1024**2:.2f} MB")
    print(f"Memory delta for op: {(mem_after - mem_before) / 1024**2:.2f} MB")

    return total_flops, mem_after - mem_before
