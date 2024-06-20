"""
FineWeb-Edu dataset (for serious pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Configuration
LOCAL_DIR = "edu_fineweb10B"
REMOTE_NAME = "sample-10BT"
SHARD_SIZE = int(1e8)  # 100M tokens per shard, total of 100 shards

# Create the cache directory if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), LOCAL_DIR)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=REMOTE_NAME, split="train")

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens['<|endoftext|>']  # End of text token

def tokenize(doc):
    """Tokenizes a single document and returns a numpy array of uint16 tokens."""
    tokens = [EOT]  # The special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    """Writes the tokenized data to a file."""
    try:
        np.save(filename, tokens_np)
    except IOError as e:
        print(f"Error writing file {filename}: {e}")

# Tokenize all documents and write output shards, each of SHARD_SIZE tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # Preallocate buffer to hold current shard
    all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # Is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < SHARD_SIZE:
            # Simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # Update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # Write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # Split the document into whatever fits in this shard; the remainder goes to the next one
            remainder = SHARD_SIZE - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # Populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # Write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])