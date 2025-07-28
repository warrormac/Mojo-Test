from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx
from sys import has_accelerator

fn print_threads():
    """Print thread IDs."""
    print("Block index: [",
        block_idx.x,
        "]\tThread index: [",
        thread_idx.x,
        "]"
    )

def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        ctx.enqueue_function[print_threads](grid_dim=2, block_dim=64)
        ctx.synchronize()
        print("Program finished")