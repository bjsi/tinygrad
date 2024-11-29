import math
from typing import Callable, Optional, Tuple
from tinygrad.dtype import dtypes
from tinygrad.engine import jit
from tinygrad.tensor import Tensor

def mix(states: Tensor, chunks: Tensor) -> Tensor:
  def rotr(x: Tensor, n: int) -> Tensor: return ((x << (32 - n)) | (x >> n))
  for i, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
    mx, my = chunks[i * 2], chunks[i * 2 + 1]
    for m in (mx, my):
      states[a] = states[a] + states[b] + m
      states[d] = rotr(states[d] ^ states[a], 16 if m is mx else 8)
      states[c] = states[c] + states[d]
      states[b] = rotr(states[b] ^ states[c], 12 if m is mx else 7)

def compress_chunks(states: Tensor, chunks: Tensor, chain_vals: Tensor, n_end_blocks: int, mix_jitted: Optional[Callable] = None):
  for i in range(16): # parallel over chunks, sequential over blocks
    compressed = compress_blocks(states[i].contiguous(), chunks[i].contiguous(), chain_vals[i].contiguous(), mix_jitted)
    if i < chunks.shape[1] - 1: states[i + 1, :8] = compressed[:8] # propagate chain vals to the next block
    if i == n_end_blocks - 1: final_chain_val = compressed[:8, -1:] # for partial chunks
  return compressed[:8] if n_end_blocks == 16 else compressed[:8, :-1].cat(final_chain_val, dim=-1)

def compress_blocks(states: Tensor, chunks: Tensor, chain_vals: Tensor, mix_jitted: Optional[Callable] = None) -> Tensor:
  mix_jitted = mix_jitted or mix
  for _ in range(6):
    mix_jitted(states, chunks)
    chunks.replace(chunks[[2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]])
  mix_jitted(states, chunks)
  states[:8] = states[:8] ^ states[8:]
  states[8:] = chain_vals[:8] ^ states[8:]
  return states

def tensor_to_blake_data(tensor: Tensor) -> Tuple[Tensor, int, int]:
  data = tensor.flatten().bitcast(dtypes.uint8)
  unpadded_len = data.numel()
  data = data.pad(((0, 1024 if data.shape[0] == 0 else (1024 - (data.shape[0] % 1024)) % 1024),), value=0)
  data = data.bitcast(dtypes.uint32).reshape(-1, 16, 16).permute(1, 2, 0).contiguous()
  final_chunk_bytes = unpadded_len - (data.shape[2] - 1) * 1024
  n_end_blocks = max(1, (final_chunk_bytes // 64) + (1 if final_chunk_bytes % 64 else 0))
  end_block_len = 64 if unpadded_len % 64 == 0 and unpadded_len else unpadded_len % 64
  return data, n_end_blocks, end_block_len # data is [blocks, words, chunks]

def pairwise_concat(chain_vals: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
  leftover_chunk = chain_vals[:, -1:] if chain_vals.shape[1] % 2 else None
  chain_vals = chain_vals[:, :-1 if leftover_chunk is not None else None]
  return chain_vals.permute(1, 0).reshape(-1, 16).transpose().contiguous(), leftover_chunk

def create_state(iv: Tensor, count: Optional[int], end_block_len: Optional[int], n_end_blocks: Optional[int], flags: Tensor) -> Tensor:
  n_blocks, _, n_chunks = iv.shape
  if count is not None:
    counts = Tensor.arange(count, count + n_chunks, dtype=dtypes.uint32).reshape(-1, 1).expand(-1, 16).reshape(-1, 16, 1)
    counts = counts.permute(1, 2, 0).cat(Tensor.zeros(n_blocks, 1, n_chunks, dtype=dtypes.uint32), dim=1)
  else:
    counts = Tensor.zeros((n_blocks, 2, n_chunks), dtype=dtypes.uint32)
  lengths = Tensor.full((n_blocks, 1, n_chunks), fill_value=64, dtype=dtypes.uint32).contiguous()
  if end_block_len is not None: lengths[n_end_blocks - 1, :, -1] = end_block_len
  return iv.cat(iv[:, :4], counts, lengths, flags, dim=1).contiguous()

def create_flags(data: Tensor, n_end_blocks: Optional[int], root: bool, parent: bool) -> Tensor:
  end_idx = n_end_blocks - 1 if n_end_blocks is not None else -1
  flags = Tensor.zeros((16, 1, data.shape[-1]), dtype=dtypes.uint32).contiguous()
  flags[0] = flags[0] + 1 # chunk start flag
  flags[-1, 0, :-1] = flags[-1, 0, :-1] + 2 # chunk end flag
  flags[end_idx:, :, -1] = flags[end_idx:, :, -1] + 2 # final chunk end flag for partial chunk
  if parent: flags[:] = 4 # parent flag
  if root: flags[end_idx, :, -1] = flags[end_idx, :, -1] + 8 # root flag
  return flags

def compress(data, compressor, count, end_block_len, n_end_blocks, root, parent, mix_jitted: Optional[Callable] = None) -> Tensor:
  init_chain_vals = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
  iv = Tensor(init_chain_vals, dtype=dtypes.uint32).reshape(1, 8, 1).expand(16, 8, data.shape[-1]).contiguous()
  states = create_state(iv, count, end_block_len, n_end_blocks, create_flags(data, n_end_blocks, root, parent))
  return compressor(states, data, iv, n_end_blocks, mix_jitted)

def blake3(tensor: Tensor) -> str:
  """Hash a Tensor in parallel using the BLAKE3 hashing algorithm."""
  data, n_end_blocks, end_block_len = tensor_to_blake_data(tensor)
  mix_jitted = jit.TinyJit(mix) if data.shape[2] > 1 else None
  chain_vals = compress(data, compress_chunks, 0, end_block_len, n_end_blocks, data.shape[-1] == 1, False, mix_jitted)
  n_steps = math.ceil(math.log2(max(chain_vals.shape[-1], 1)))
  tree_compressor = lambda states, data, iv, _, mix_jitted: compress_blocks(states[-1].contiguous(), data, iv[0].contiguous(), mix_jitted)
  for i in range(n_steps): # tree-hash chain value pairs ~halving them in each step
    chain_vals, leftover_chain_val = pairwise_concat(chain_vals)
    pre_pad_size = chain_vals.shape[1] # use padding to keep the same batch dim - faster
    if i < n_steps - 1: chain_vals = chain_vals.pad(((0,0), (0, data.shape[2] - pre_pad_size)), value=0).contiguous()
    chain_vals = compress(chain_vals, tree_compressor, None, None, None, i == n_steps - 1, True, mix_jitted if i < n_steps - 1 else None)
    chain_vals = chain_vals[:8, :pre_pad_size if i < n_steps - 1 else None]
    if leftover_chain_val is not None: chain_vals = chain_vals.cat(leftover_chain_val, dim=1)
  return chain_vals[:, 0].flatten().bitcast(dtypes.uint8).data().tobytes()[:32].hex()

if __name__ == "__main__":
  import time
  t = Tensor.ones(1024 * 1024 * 2000 , dtype=dtypes.uint8)
  st = time.time()
  print(blake3(t))
  print(f"Hashed 1GB in {time.time()-st:.2f}s")
