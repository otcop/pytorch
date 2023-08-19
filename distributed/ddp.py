#From https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#skewed-processing-speeds
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
    os.environ['MASTER_PORT'] = '12356'
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

    CHECKPOINT_PATH =  "./model.checkpoint"
    print(rank)
    if rank == 0:
        print("load model")
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

#     if rank == 0:
#         os.remove(CHECKPOINT_PATH)
#     cleanup()

class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 1).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)

def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel")
    dev0 = rank*2
    dev1 = rank*2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ddp_mp_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20,10))
    labels = torch.randn(20,1).to(dev0)
    loss = loss_fn(outputs, labels)
    loss.backward()
    print(loss.item())
    optimizer.step()

if __name__ =="__main__":
    run_demo(demo_checkpoint, 2)