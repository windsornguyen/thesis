import torch
import torch.nn.functional as F
import tiktoken

from time import time
from tqdm import tqdm
from model.stu import SSSMConfigs, SpectralStateSpaceModel

device = ('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {device}')

# Load the checkpoint
print('Loading the checkpoint...')
start_time = time()
checkpoint = torch.load('inference/model_00634.pt', map_location=device)
print(f'Successfully loaded the checkpoint in {time() - start_time:.2f} seconds')

configs = SSSMConfigs(
    n_embd=768,
    block_size=1_024,
    vocab_size=50_304,
    num_layers=6,
    dropout=0.10,
    input_len=1_024,
    scale=4,
    bias=True,
    num_eigh=24,
    auto_reg_k_u=3,
    auto_reg_k_y=2,
    learnable_m_y=True,
)
model = SpectralStateSpaceModel(configs)
model = torch.compile(model)

# Load the saved states into the model
model.load_state_dict(checkpoint['model'])
model.to(device)
print(
    f"Successfully loaded the model from step {checkpoint['step']} with validation loss {checkpoint['val_loss']}"
)
model.eval()

# Prepare generation
tokenizer = tiktoken.get_encoding('gpt2')
num_return_sequences = 5
max_length = 16
prompt = "Hi, I'm a language model,"
tokens = tokenizer.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)

print(
    f"\nGenerating {num_return_sequences} sequences of maximum length {max_length} based on the prompt: '{prompt}'"
)

with torch.no_grad():
    with tqdm(total=max_length - xgen.size(1), desc='Generating') as pbar:
        while xgen.size(1) < max_length:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(xgen)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(
                topk_probs, 1, generator=sample_rng
            )  # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            xgen = torch.cat((xgen, xcol), dim=1)
            pbar.update(1)

# Print the generated text
print()
for i in range(num_return_sequences):
    tokens = xgen[i, :].tolist()
    decoded = tokenizer.decode(tokens)
    print(f'Sample {i+1}: {decoded}')
    print()
