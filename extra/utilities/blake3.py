import math
from typing import Callable, Optional, Tuple
from tinygrad.dtype import dtypes
from tinygrad.engine import jit
from tinygrad.tensor import Tensor

# try adding masks
# [x] fix chunks.replace

def mix(states: Tensor, chunks: Tensor) -> Tensor:
  print(states.shape, chunks.shape)
  def rotr(x: Tensor, n: int) -> Tensor: return ((x << (32 - n)) | (x >> n))
  for i, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
    mx, my = chunks[:, i * 2], chunks[:, i * 2 + 1]
    # First iteration with mx
    new_a = states[:, a]
    new_b = states[:, b]
    new_c = states[:, c]
    new_d = states[:, d]

    new_a = (new_a + new_b + mx).realize()
    new_d = rotr(new_d ^ new_a, 16)
    new_c = (new_c + new_d)
    new_b = rotr(new_b ^ new_c, 12)

    # Second iteration with my
    new_a = (new_a + new_b + my)
    new_d = rotr(new_d ^ new_a, 8)
    new_c = (new_c + new_d)
    new_b = rotr(new_b ^ new_c, 7)

    states[:, a] = new_a
    states[:, b] = new_b
    states[:, c] = new_c
    states[:, d] = new_d
  return states.realize()

def compress_chunks(states: Tensor, chunks: Tensor, chain_vals: Tensor, n_end_blocks: int):
  print("-" * 100)
  print("CHUNKS START")
  print("-" * 100)
  for i in range(16): # parallel over chunks, sequential over blocks
    print("chunks iteration", i)
    compressed = compress_blocks(states[:, i].contiguous(), chunks[:, i].contiguous(), chain_vals[:, i].contiguous())
    if i < chunks.shape[1] - 1: states[:, i + 1, :8] = compressed[:, :8] # propagate chain vals to the next block
  end_idx = n_end_blocks - 1 if chunks.shape[0] == 1 else n_end_blocks # handle partial final chunk
  print("CHUNKS DONE")
  print("-" * 100)
  return compressed[:, :8] if n_end_blocks == 16 else compressed[:-1, :8].cat(states[-1:, end_idx, :8])

def compress_blocks(states: Tensor, chunks: Tensor, chain_vals: Tensor) -> Tensor:
  print("-" * 100)
  print("BLOCKS START")
  print("-" * 100)
  mix_jit = jit.TinyJit(mix)
  for _ in range(6):
    print("blocks iteration", _)
    mix_jit(states, chunks)
    chunks = chunks[:, [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]]
  mix_jit(states, chunks)
  states[:, :8] = states[:, :8] ^ states[:, 8:]
  states[:, 8:] = chain_vals[:, :8] ^ states[:, 8:]
  return states

def tensor_to_blake_chunks(tensor: Tensor) -> Tuple[Tensor, int, int]:
  data = tensor.flatten().bitcast(dtypes.uint8)
  unpadded_len = data.numel()
  data = data.pad(((0, 1024 if data.shape[0] == 0 else (1024 - (data.shape[0] % 1024)) % 1024),), value=0)
  data = data.bitcast(dtypes.uint32).reshape(-1, 16, 16)
  final_chunk_bytes = unpadded_len - (data.shape[0] - 1) * 1024
  n_end_blocks = max(1, (final_chunk_bytes // 64) + (1 if final_chunk_bytes % 64 else 0))
  end_block_len = 64 if unpadded_len % 64 == 0 and unpadded_len else unpadded_len % 64
  return data, n_end_blocks, end_block_len # data is [chunks, blocks, words]

def pairwise_concatenate(chain_vals: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
  leftover_chunk = chain_vals[-1:] if chain_vals.shape[0] % 2 else None
  return chain_vals[:-1 if leftover_chunk is not None else None].reshape(-1, 16), leftover_chunk

def create_state(iv: Tensor, count: Optional[int], end_block_len: Optional[int], n_end_blocks: Optional[int], flags: Tensor) -> Tensor:
  n_chunks, n_blocks = iv.shape[0], iv.shape[1]
  if count is not None:
    counts = Tensor.arange(count, count + n_chunks, dtype=dtypes.uint32).reshape(-1, 1).expand(-1, 16).reshape(-1, 16, 1)
    counts = counts.cat(Tensor.zeros(n_chunks, n_blocks, 1, dtype=dtypes.uint32), dim=-1)
  else:
    counts = Tensor.zeros(n_chunks, n_blocks, 2, dtype=dtypes.uint32)
  lengths = Tensor.full((n_chunks, n_blocks, 1), fill_value=64, dtype=dtypes.uint32).contiguous()
  if end_block_len is not None: lengths[-1, n_end_blocks - 1] = end_block_len
  return iv.cat(iv[:, :, :4], counts, lengths, flags, dim=-1).contiguous()

def create_flags(chunks: Tensor, n_end_blocks: Optional[int], root: bool, parent: bool) -> Tensor:
  end_idx = n_end_blocks - 1 if n_end_blocks is not None else -1
  flags = Tensor.full((chunks.shape[0], chunks.shape[1], 1), fill_value=4 if parent else 0, dtype=dtypes.uint32).contiguous()
  flags[:, 0] += flags[:, 0] + 1 # chunk start flag
  flags[:-1, -1] = flags[:-1, -1] + 2 # chunk end flag
  flags[-1, end_idx:] = flags[-1, end_idx:] + 2 # chunk end flag for partial final chunk
  if root: flags[0, end_idx] = flags[0, end_idx] + 8 # root flag
  return flags

def compress(data, compressor, count, end_block_len, n_end_blocks, root, parent) -> Tensor:
  iv = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)
  iv = iv.expand(data.shape[0], data.shape[1], -1).contiguous()
  states = create_state(iv, count, end_block_len, n_end_blocks, create_flags(data, n_end_blocks, root, parent))
  return compressor(states, data, iv, n_end_blocks)[:, :8]

def blake3(tensor: Tensor) -> str:
  """Hash a Tensor in parallel using the BLAKE3 hashing algorithm."""
  chunks, n_end_blocks, end_block_len = tensor_to_blake_chunks(tensor)
  chain_vals = compress(chunks, compress_chunks, 0, end_block_len, n_end_blocks, chunks.shape[0] == 1, False)
  tree_levels = math.ceil(math.log2(max(chain_vals.shape[0], 1)))
  tree_compressor = lambda states, data, iv, _: compress_blocks(states[:, -1].contiguous(), data, iv[:, 0])
  for i in range(tree_levels): # tree-hash chain value pairs ~halving the number in each step
    chain_vals, leftover_chain_val = pairwise_concatenate(chain_vals)
    chain_vals = compress(chain_vals, tree_compressor, None, None, None, i == tree_levels - 1, True)
    if leftover_chain_val is not None: chain_vals = chain_vals.cat(leftover_chain_val, dim=0)
  return chain_vals[0].flatten().bitcast(dtypes.uint8).data().tobytes()[:32].hex()


if __name__ == "__main__":
  data = 1024 * 1024 * 1000
  input = Tensor.full((data,), fill_value=0x41, dtype=dtypes.uint8)
  import time
  start = time.monotonic()
  result = blake3(input)
  duration = time.monotonic() - start
  print(f"Hashed {data/1024/1024:.1f}MB in {duration:.3f}s ({data/duration/1024/1024:.1f}MB/s)")
  print(f"Hash: {result}")
