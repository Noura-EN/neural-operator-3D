"""
HPC environment detection and distributed training setup utilities.
Supports PBSPro workload manager.
"""

import os
import socket
import torch
import torch.distributed as dist


def detect_hpc_environment():
    """
    Detect if running in HPC environment (PBSPro).
    
    Returns:
        bool: True if HPC environment detected
    """
    return os.environ.get('PBS_NODEFILE') is not None


def parse_pbs_nodefile(nodefile_path: str = None):
    """
    Parse PBS nodefile to get node list.
    
    Args:
        nodefile_path: Path to PBS nodefile (defaults to $PBS_NODEFILE)
    
    Returns:
        List of node hostnames
    """
    if nodefile_path is None:
        nodefile_path = os.environ.get('PBS_NODEFILE')
    
    if nodefile_path is None or not os.path.exists(nodefile_path):
        return None
    
    with open(nodefile_path, 'r') as f:
        nodes = [line.strip() for line in f if line.strip()]
    
    return nodes


def setup_distributed(
    backend: str = "nccl",
    master_addr: str = None,
    master_port: str = "29500"
):
    """
    Setup distributed training environment.
    
    Args:
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)
        master_addr: Master address (auto-detected if None)
        master_port: Master port
    
    Returns:
        Tuple of (rank, world_size, local_rank) or (None, None, None) if not distributed
    """
    # Check if already initialized
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        return rank, world_size, local_rank
    
    # Check for environment variables (set by torchrun)
    rank = os.environ.get('RANK')
    world_size = os.environ.get('WORLD_SIZE')
    local_rank = os.environ.get('LOCAL_RANK')
    
    if rank is not None:
        # Using torchrun
        rank = int(rank)
        world_size = int(world_size)
        local_rank = int(local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr or 'localhost'}:{master_port}",
            rank=rank,
            world_size=world_size
        )
        
        return rank, world_size, local_rank
    
    # Check for PBSPro environment
    if detect_hpc_environment():
        nodes = parse_pbs_nodefile()
        if nodes is None or len(nodes) == 0:
            return None, None, None
        
        # Determine master address
        if master_addr is None:
            master_addr = nodes[0].split('.')[0]  # Remove domain if present
        
        # Get current hostname
        hostname = socket.gethostname().split('.')[0]
        
        # Find rank based on position in nodefile
        try:
            rank = nodes.index(hostname)
        except ValueError:
            # Hostname not in nodefile, might be running locally
            return None, None, None
        
        world_size = len(nodes)
        
        # Local rank is position within node (simplified - assumes 1 GPU per node)
        local_rank = 0
        
        # Initialize process group
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size
        )
        
        return rank, world_size, local_rank
    
    # Not in distributed environment
    return None, None, None


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(rank: int = None, local_rank: int = None):
    """
    Get appropriate device for current process.
    
    Args:
        rank: Process rank (for logging)
        local_rank: Local rank (for device assignment)
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if local_rank is not None:
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    return device


def is_master_process(rank: int = None):
    """
    Check if current process is master (rank 0).
    
    Args:
        rank: Process rank (None means single process)
    
    Returns:
        bool: True if master process
    """
    if rank is None:
        return True
    return rank == 0
