# ==============================================================================#
# Authors: Windsor Nguyen
# File: model.py
# ==============================================================================#

"""Spectral temporal unit (STU) block."""

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from model import stu_utils


@dataclass
class SSSMConfigs:
    n_embd: int = 384
    block_size: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    dropout: float = 0.1
    input_len: int = 1024
    scale: int = 4
    bias: bool = True
    num_eigh: int = 256
    auto_reg_k_u: int = 3
    auto_reg_k_y: int = 2
    learnable_m_y: bool = True  # TODO: Needed?


class STU(nn.Module):
    """
    A simple STU (Spectral Transform Unit) Layer.

    Args:
        d_out (int): Output dimension.
        input_len (int): Input sequence length.
        num_eigh (int): Number of eigenvalues and eigenvectors to use.
        auto_reg_k_u (int): Auto-regressive depth on the input sequence.
        auto_reg_k_y (int): Auto-regressive depth on the output sequence.
        learnable_m_y (bool): Whether the m_y matrix is learnable.
    """

    def __init__(
        self,
        d_out: int = 24,
        input_len: int = 1000,
        num_eigh: int = 24,
        auto_reg_k_u: int = 3,
        auto_reg_k_y: int = 2,
        learnable_m_y: bool = True,
    ) -> None:
        super(STU, self).__init__()
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.d_out = d_out
        self.l, self.k = input_len, num_eigh
        self.eigh = stu_utils.get_top_hankel_eigh(self.l, self.k, device=self.device)
        self.auto_reg_k_u = auto_reg_k_u
        self.auto_reg_k_y = auto_reg_k_y
        self.learnable_m_y = learnable_m_y
        self.m_phi = nn.Parameter(torch.empty([self.d_out * self.k, self.d_out]))

        # Matrices for autoregressive stuff
        # self.m_x = 1.0 / (float(self.d_out) ** 0.5)
        # self.m_u = nn.Parameter(
        #     torch.empty([self.d_out, self.d_out, self.auto_reg_k_u])
        # )
        # self.m_y = (
        #     nn.Parameter(torch.empty([self.d_out, self.auto_reg_k_y, self.d_out]))
        #     if learnable_m_y
        #     else self.register_buffer(
        #         "m_y", torch.empty([self.d_out, self.auto_reg_k_y, self.d_out])
        #     )
        # )

    def apply_stu(self, inputs):
        eig_vals, eig_vecs = self.eigh
        x_tilde = stu_utils.compute_x_tilde(inputs, (eig_vals, eig_vecs))
        delta_phi = x_tilde @ self.m_phi
        return delta_phi

    def forward(self, inputs):
        output = self.apply_stu(inputs)
        return output


class FFN(nn.Module):
    """
    Simple feed-forward network.
    """

    def __init__(self, config):
        super(FFN, self).__init__()
        self.c_fc = nn.Linear(
            config.n_embd, config.scale * config.n_embd, bias=config.bias
        )
        # TODO: Consider implementing Squared ReLU from https://arxiv.org/pdf/2109.08668
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            config.scale * config.n_embd, config.n_embd, bias=config.bias
        )
        self.c_proj.SSSM_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class STUBlock(nn.Module):
    def __init__(self, config):
        super(STUBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.stu = STU(
            d_out=config.n_embd,
            input_len=config.input_len,
            num_eigh=config.num_eigh,
            auto_reg_k_u=config.auto_reg_k_u,
            auto_reg_k_y=config.auto_reg_k_y,
            learnable_m_y=config.learnable_m_y,
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = FFN(config)

    def forward(self, x):
        x = x + self.stu(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class SpectralStateSpaceModel(nn.Module):
    """
    General language model architecture based on STU blocks.
    """

    def __init__(self, config):
        super(SpectralStateSpaceModel, self).__init__()
        self.config = config
        self.std = config.n_embd**-0.5
        self.stu_block = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                hidden=nn.ModuleList(
                    [STUBlock(self.config) for _ in range(config.num_layers)]
                ),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        # Weight sharing scheme
        self.stu_block.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        print("STU Model Parameter Count: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, idx, tgts=None):
        # idx is of shape (bsz, sl)
        bsz, sl = idx.size()
        assert (
            sl <= self.config.block_size
        ), f"Cannot forward sequence of length {sl}, block size is only {self.config.block_size}"
        pos = torch.arange(0, sl, dtype=torch.long, device=idx.device)
        pos_emb = self.stu_block.wpe(pos)  # (sl, n_embd)
        tok_emb = self.stu_block.wte(idx)  # (bsz, sl, n_embd)
        x = tok_emb + pos_emb  # (bsz, sl, n_embd)
        for block in self.stu_block.hidden:
            x = block(x)
        x = self.stu_block.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if tgts is not None:
            # Flatten logits and compute the cross-entropy loss
            loss = F.cross_entropy(
                logits.view(bsz * sl, logits.size(-1)), tgts.view(bsz * sl)
            )
        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, "SSSM_SCALE_INIT"):
                self.std *= (2 * self.config.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, STU):
            torch.nn.init.xavier_normal_(module.m_phi)
            # m_x = 1.0 / (float(module.d_out) ** 0.5)
            # torch.nn.init.uniform_(module.m_u, -m_x, m_x)
            # if module.learnable_m_y:
            #     torch.nn.init.xavier_normal_(module.m_y)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.

        Args:
            non_embedding (bool, optional):
            Whether to exclude the positional embeddings (if applicable).
            Defaults to True.

        Returns:
            int: The number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(
                f"Optimizer | Number of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"Optimizer | Number of non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"Optimizer | Using fused AdamW?: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, D, E, T = cfg.num_layers, cfg.n_embd, cfg.num_eigh, cfg.input_len
        
        # Embedding layers
        embed_flops = 2 * D * T

        # STU blocks
        stu_block_flops = 0
        for _ in range(L):
            # Layer normalization
            stu_block_flops += 2 * D * T  # ln_1 and ln_2
            
            # STU layer
            stu_block_flops += 2 * E * D * T  # Compute x_tilde
            stu_block_flops += 2 * D * E * D  # Apply m_phi matrix
            
            # FFN layer
            stu_block_flops += 2 * D * cfg.scale * D  # c_fc
            stu_block_flops += cfg.scale * D  # GELU activation
            stu_block_flops += 2 * cfg.scale * D * D  # c_proj
        
        # Final layer normalization
        final_ln_flops = 2 * D * T  # ln_f
        
        # Language model head
        lm_head_flops = 2 * D * cfg.vocab_size
        
        flops_per_iter = embed_flops + stu_block_flops + final_ln_flops + lm_head_flops
        flops_per_fwdbwd = flops_per_iter * fwdbwd_per_iter
        
        # Express flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_fwdbwd / dt  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


    def flops_per_token(self):
        """Estimate the number of floating-point operations per token."""
        flops = 0
        cfg = self.config

        # Embedding layers
        flops += 2 * cfg.n_embd * cfg.block_size  # wte and wpe embeddings

        # STU blocks
        for _ in range(cfg.num_layers):
            # Layer normalization
            flops += 2 * cfg.n_embd * cfg.block_size  # ln_1 and ln_2

            # STU layer
            flops += 2 * cfg.num_eigh * cfg.n_embd * cfg.block_size  # Compute x_tilde
            flops += 2 * cfg.n_embd * cfg.num_eigh * cfg.n_embd  # Apply m_phi matrix

            # FFN layer
            flops += 2 * cfg.n_embd * cfg.scale * cfg.n_embd  # c_fc
            flops += cfg.scale * cfg.n_embd  # GELU activation
            flops += 2 * cfg.scale * cfg.n_embd * cfg.n_embd  # c_proj

        # Final layer normalization
        flops += 2 * cfg.n_embd * cfg.block_size  # ln_f

        # Language model head
        flops += 2 * cfg.n_embd * cfg.vocab_size

        return flops

    @torch.no_grad()
    def inference(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at ctxt_len
            idx_cond = (
                idx
                if idx.size(1) <= self.config.ctxt_len
                else idx[:, -self.config.ctxt_len :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
