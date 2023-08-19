# From PyTorch Website https://pytorch.org/tutorials/intermediate/ddp_tutorial.html?utm_source=distr_landing&utm_medium=intermediate_ddp_tutorial
import os 
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import tempfile

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10,10)
        self.l2 = nn.Linear(10,1)
    def forward(self,x):
        return self.l2(torch.relu(self.l1(x)))

def demo_basic(rank, world_size):
    setup(rank, world_size)
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20,10))
    labels = torch.randn(20,1).to(rank)
    loss = loss_fn(outputs, labels)
    loss.backward()
    print(loss.item())
    optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args = (world_size,),
             nprocs = world_size,
             join=True)

def demo_checkpoint(rank, world_size):
    setup(rank, world_size)
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)
    
    dist.barrier()
    map_location = {"cuda:%d" % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location)
    )
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20,10))
    labels = torch.randn(20,1).to(rank)
    loss = loss_fn(outputs, labels)
    loss.backward()
    print(loss.item())
    optimizer.step()

    if rank == 0:
        os.remove(CHECKPOINT_PATH)
    cleanup()
if __name__ =="__main__":
    run_demo(demo_basic, 2)