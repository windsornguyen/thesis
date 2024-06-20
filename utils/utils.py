import math

def get_lr(it, warmup_iters, max_steps, max_lr, min_lr):
    # 1. Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters

    # 2. If it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr

    # 3. If in between, cosine decay to down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_steps - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1 + math.cos(math.pi * decay_ratio)
    )  # Coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)
