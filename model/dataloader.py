import torch
import tiktoken  # TODO: Write the custom one from Llama 3 repository
import numpy as np
import os

def load_tokens(file_name):
    npt = np.load(file_name)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class Dataloader:
    def __init__(self, bsz, sl, process_rank, num_processes, dataset, split):
        self.bsz = bsz # batch size
        self.sl = sl # sequence length
        self.process_rank = process_rank
        self.master_process = self.process_rank == 0
        self.num_processes = num_processes
        assert split in {'train', 'val', 'test'}

        # Get the shard filenames
        data_root = dataset
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f'no shards found for split {split}'
        if self.master_process:
            print(f'found {len(shards)} shards for split {split}')
        self.reset()
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.idx = self.bsz * self.sl * self.process_rank

    def next_batch(self):
        bsz, sl = self.bsz, self.sl
        buf = self.tokens[self.idx : self.idx + bsz * sl + 1]
        X = (buf[:-1]).view(bsz, sl)
        y = (buf[1:]).view(bsz, sl)
        self.idx += bsz * sl * self.num_processes
        
        if self.idx + (bsz * sl * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.idx = self.bsz * self.sl * self.process_rank

        return X, y
