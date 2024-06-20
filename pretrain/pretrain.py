# ==============================================================================#
# Authors: Windsor Nguyen
# File: train.py
# ==============================================================================#

"""Distributed training and testing for an STU-based language model."""

import argparse
import os
import time
import math

import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from model.stu import SpectralStateSpaceModel, SSSMConfigs
from model.transformer import Transformer, TransformerConfigs
from model.dataloader import Dataloader
from dist_utils import setup, cleanup
from utils import get_lr
from tqdm import tqdm
from hellaswag import iterate_examples, render_example, get_most_likely_row

def main():
    # Distributed setup
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('SLURM_PROCID', 0))
    gpus_per_node = int(os.environ.get('SLURM_GPUS_ON_NODE', 0))
    device, local_rank, world_size = setup(rank, world_size, gpus_per_node)
    ddp = world_size > 1
    master_process = local_rank == 0

    torch.manual_seed(1337 + local_rank)

    print(f"Using device: {device}")


    # Shared hyperparameters
    block_size = 1024
    vocab_size = 50_304
    num_layers = 12
    dropout = 0.10
    scale = 4

    # SSSM-specific hyperparameters
    n_embd_sssm = 384
    input_len = 1_024
    bias = True
    num_eigh = 24
    auto_reg_k_u = 3
    auto_reg_k_y = 2
    learnable_m_y = True

    # Transformer-specific hyperparameters
    n_embd_transformer = 384
    n_head = 12

    # Model sampling parameters
    num_return_sequences = 4
    max_length = 32

    # Create model
    parser = argparse.ArgumentParser(description='Choose the architecture to train.')
    parser.add_argument('--model', type=str, default='stu', choices=['stu', 'transformer'], help='Specify the model type: "stu" or "transformer"')
    args = parser.parse_args()

    # Configure the model
    if args.model == 'stu':
        configs = SSSMConfigs(
            n_embd=n_embd_sssm,
            block_size=block_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            dropout=dropout,
            input_len=input_len,
            scale=scale,
            bias=bias,
            num_eigh=num_eigh,
            auto_reg_k_u=auto_reg_k_u,
            auto_reg_k_y=auto_reg_k_y,
            learnable_m_y=learnable_m_y,
        )
        model = SpectralStateSpaceModel(configs)
    else: 
        config = TransformerConfigs(
            block_size=block_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            n_head=n_head,
            n_embd=n_embd_transformer,
            dropout=dropout,
            scale=scale,
        )
        model = Transformer(config)

    model.eval()
    model.to(device)

    # TODO: Fix this: torch.compile interferes with HellaSwag eval and Generation.
    use_compile = True
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    total_params = sum(p.numel() for p in model.parameters())


    # Gradient accumulation section
    total_bsz = 524_288  # 2 ** 19, ~0.5M, in number of tokens
    bsz = 32  # micro batch size
    sl = 1_024  # sequence length

    assert (
        total_bsz % (bsz * sl * world_size) == 0
    ), f"total_bsz ({total_bsz}) must be divisible by bsz * sl * world_size ({bsz * sl * world_size})"
    grad_accum_steps = total_bsz // (bsz * sl * world_size)

    if master_process:
        print(f"Total (desired) batch size: {total_bsz}")
        print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")

    # Data loader section
    dataset = 'data/edu_fineweb10B'
    train_loader = Dataloader(bsz=bsz, sl=sl, process_rank=local_rank, num_processes=world_size, dataset=dataset, split='train')
    val_loader = Dataloader(bsz=bsz, sl=sl, process_rank=local_rank, num_processes=world_size, dataset=dataset, split='val')

    # Easy optimization section
    torch.set_float32_matmul_precision("high")  # Enable tf-32

    # Set scale in case we do smaller runs
    scale = 30

    # Learning rate used for GPT-3 Small in the GPT-3 paper
    max_lr = 6e-4
    min_lr = max_lr * 0.1

    # Per the GPT-3 paper, the warmup phase occurs over the first 375 million tokens.
    # => 375e6 / (2**19) ≈ 715 steps.
    warmup_iters = 715 // scale

    num_epochs = 1

    # We are training on 10B tokens with 2^19 tokens per step
    # => 10e9 / (2**19) ≈ 19,073 steps.
    # max_steps = num_epochs * 19_073
    max_steps = num_epochs * 19_073 // scale
    print(f'Training on {max_steps} steps')

    # Optimizer and learning rate scheduler section
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, learning_rate=max_lr, device_type=device.type, master_process=master_process
    )

    # create the log directory we will write checkpoints to and log to
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass


    # Train!
    pbar = tqdm(range(max_steps), desc='Training', unit='step', dynamic_ncols=True)
    for step in pbar:
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # Evaluate the model every once in a while
        if step % (250 // scale) == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in tqdm(range(val_loss_steps), desc='Validation', unit='batch'):
                    X, y = val_loader.next_batch()
                    X, y = X.to(device), y.to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        logits, loss = model(X, y)
                    loss /= val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                # TODO: Use pbar write or print statement?
                # pbar.write(f"Validation loss: {val_loss_accum.item():.4f}")
                print(
                    f"Validation loss: {val_loss_accum.item():.4f}"
                )
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % (5000 // scale) == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'optimizer':  optimizer.state_dict(),
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                        'rng_state_pytorch': torch.get_rng_state(),
                        'rng_state_cuda': torch.cuda.get_rng_state(),
                        'rng_state_numpy': np.random.get_state()
                    }
                    torch.save(checkpoint, checkpoint_path)


        # Periodically evaluate on HellaSwag
        # if (step % 250 == 0 or last_step) and (not use_compile):
        #     num_correct_norm = 0
        #     num_total = 0
        #     for i, example in tqdm(enumerate(iterate_examples("val")), desc='Evaluating on HellaSwag', unit='example', total=10042, dynamic_ncols=True):
        #         # only process examples where i % ddp_world_size == rank
        #         if i % world_size != rank:
        #             continue
        #         # render the example into tokens and labels
        #         _, tokens, mask, label = render_example(example)
        #         tokens = tokens.to(device)
        #         mask = mask.to(device)
        #         # get the logits
        #         with torch.no_grad():
        #             with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        #                 logits, loss = model(tokens)
        #             pred_norm = get_most_likely_row(tokens, mask, logits)
        #         num_total += 1
        #         num_correct_norm += int(pred_norm == label)
        #     # reduce the stats across all processes
        #     if ddp:
        #         num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        #         num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        #         dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        #         dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        #         num_total = num_total.item()
        #         num_correct_norm = num_correct_norm.item()
        #     acc_norm = num_correct_norm / num_total
        #     if master_process:
        #         print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        #         with open(log_file, "a") as f:
        #             f.write(f"{step} hella {acc_norm:.4f}\n")

        # once in a while generate from the model (except step 0, which is noise)
        # if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        #     model.eval()
        #     tokens = enc.encode("Hello, I'm a language model,")
        #     tokens = torch.tensor(tokens, dtype=torch.long)
        #     tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        #     xgen = tokens.to(device)
        #     sample_rng = torch.Generator(device=device)
        #     sample_rng.manual_seed(42 + local_rank)
        #     while xgen.size(1) < max_length:
        #         # forward the model to get the logits
        #         with torch.no_grad():
        #             with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        #                 logits, loss = model(xgen) # (B, T, vocab_size)
        #             # take the logits at the last position
        #             logits = logits[:, -1, :] # (B, vocab_size)
        #             # get the probabilities
        #             probs = F.softmax(logits, dim=-1)
        #             # do top-k sampling of 50 (huggingface pipeline default)
        #             # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        #             topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        #             # select a token from the top-k probabilities
        #             # note: multinomial does not demand the input to sum to 1
        #             ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
        #             # gather the corresponding indices
        #             xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        #             # append to the sequence
        #             xgen = torch.cat((xgen, xcol), dim=1)
                    
        #     # print the generated text
        #     for i in range(num_return_sequences):
        #         tokens = xgen[i, :max_length].tolist()
        #         decoded = enc.decode(tokens)
        #         print(f"rank {rank} sample {i}: {decoded}")
                
        model.train()   
        optimizer.zero_grad()
        loss_accum = 0.0

        # Gradient accumulation
        for micro_step in range(grad_accum_steps):
            X, y = train_loader.next_batch()
            X, y = X.to(device), y.to(device)

            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            # bfloat16 only works if using NVIDIA Ampere GPUs
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, loss = model(X, y)

            loss /= grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

            # TODO: Make sure the estimate_mfu function works for STU configs
            # fwd_bwd_per_iter = 2  # Assuming each step involves one forward and one backward pass
            # dt = t1 - t0  # Time for one iteration

            # # Compute FLOPs per token
            # if master_process:
            #     mfu = raw_model.estimate_mfu(fwd_bwd_per_base_iter, dt)
            #     print(f"Model Flops Utilization (MFU): {mfu:.4f}")

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Clip global norm of gradients to 1.0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, warmup_iters, max_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.bsz * train_loader.sl * grad_accum_steps * world_size
        tokens_per_sec = tokens_processed / dt
        # TODO: Add flops_per_token measure to the model
        # flops_per_token = raw_model.flops_per_token()
        # if master_process:
        #     print(
        #         f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | flops/tok: {flops_per_token:.2e}"
        #    )
        if master_process:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
           )
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
        # TODO: Use postfix or print statements with pretty |'s?
        # pbar.set_postfix(
        #     loss=f'{loss_accum.item():.6f}',
        #     lr=f'{lr:.4e}',
        #     norm=f'{norm:.4f}',
        #     dt=f'{dt*1000:.2f}ms',
        #     tok_per_sec=f'{tokens_per_sec:.2f}'
        # )

    cleanup()

if __name__ == '__main__':
    main()
