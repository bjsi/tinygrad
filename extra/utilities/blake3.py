import math
import os
from typing import List, Optional, Tuple
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

def mix(word_list: List[Tensor], data: Tensor):
  def rotr(x: Tensor, n: int) -> Tensor: return ((x << (32 - n)) | (x >> n))
  for i, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
    mx, my = data[i * 2], data[i * 2 + 1]
    for m in (mx, my):
      word_list[a] = (word_list[a] + word_list[b] + m).realize()
      word_list[d] = rotr(word_list[d] ^ word_list[a], 16 if m is mx else 8).realize()
      word_list[c] = (word_list[c] + word_list[d]).realize()
      word_list[b] = rotr(word_list[b] ^ word_list[c], 12 if m is mx else 7).realize()

def compress_chunks(states: Tensor, data: Tensor, chain_vals: Tensor, n_end_blocks: int):
  """Compute over tensors, storing intermediate tensors in lists."""
  word_list: List[Tensor] = [states[0, i] for i in range(8)] # init with first 8 words of the first block
  end_idx = n_end_blocks - 1 if data.shape[2] == 1 else n_end_blocks # to handle partial final chunk
  for i in range(states.shape[0]): # parallel over words, sequential over blocks
    word_list = word_list[:8] + [states[i, j] for j in range(8, 16)] # propagate chain vals to the next block
    word_list = compress_blocks(word_list, data[i], chain_vals[i])
    if i == end_idx: final_chunk_compressed = word_list # save a reference to the end idx block
  # TODO: across all chunks, until the last 
  for word in word_list: print(word.numpy())
  return word_list[:8] if n_end_blocks == 16 else word_list[:-1, :8].cat(states[-1:, end_idx, :8])

def compress_blocks(word_list: List[Tensor], data: Tensor, chain_vals: Tensor) -> List[Tensor]:
  for _ in range(6): # mix and permute
    mix(word_list, data)
    data.replace(data[[2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]])
  mix(word_list, data)
  for i in range(8):
    word_list[i] ^= word_list[i + 8]
    word_list[i + 8] ^= chain_vals[i]
  return word_list

def tensor_to_blake_data(tensor: Tensor) -> Tuple[Tensor, int, int]:
  data = tensor.flatten().bitcast(dtypes.uint8)
  unpadded_len = data.numel()
  data = data.pad(((0, 1024 if data.shape[0] == 0 else (1024 - (data.shape[0] % 1024)) % 1024),), value=0)
  data = data.bitcast(dtypes.uint32).reshape(16, 16, -1)
  final_chunk_bytes = unpadded_len - (data.shape[2] - 1) * 1024
  # TODO: replace both with simple n_bytes in final chunk
  n_end_blocks = max(1, (final_chunk_bytes // 64) + (1 if final_chunk_bytes % 64 else 0))
  end_block_len = 64 if unpadded_len % 64 == 0 and unpadded_len else unpadded_len % 64
  return data, n_end_blocks, end_block_len # data is [blocks, words, chunks]

def pairwise_concatenate(chain_vals: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
  leftover_chunk = chain_vals[-1:] if chain_vals.shape[0] % 2 else None
  return chain_vals[:-1 if leftover_chunk is not None else None].reshape(-1, 16), leftover_chunk

def create_state(iv: Tensor, count: Optional[int], end_block_len: Optional[int], n_end_blocks: Optional[int], flags: Tensor) -> Tensor:
  n_blocks, _, n_chunks = iv.shape
  if count is not None:
    counts = Tensor.arange(0, n_chunks, dtype=dtypes.uint32).reshape(1, 1, -1).expand(16, 1, -1)
    counts = counts.cat(Tensor.zeros((n_blocks, 1, n_chunks), dtype=dtypes.uint32), dim=1)
  else:
    counts = Tensor.zeros((n_blocks, 2, n_chunks), dtype=dtypes.uint32)
  lengths = Tensor.full((n_blocks, 1, n_chunks), fill_value=64, dtype=dtypes.uint32).contiguous()
  # last chunk, last block
  if end_block_len is not None: lengths[n_end_blocks - 1, 0, -1] = end_block_len
  return iv.cat(iv[:, :4], counts, lengths, flags, dim=1).contiguous()

def create_flags(data: Tensor, n_end_blocks: Optional[int], root: bool, parent: bool) -> Tensor:
  end_idx = n_end_blocks - 1 if n_end_blocks is not None else -1
  flags = Tensor.zeros((16, 1, data.shape[2]), dtype=dtypes.uint32).contiguous()
  flags[0] += 1 # chunk start flag
  flags[-1, 0, :-1] = flags[-1, 0, :-1] + 2 # chunk end flag
  flags[end_idx, 0, -1] = flags[end_idx, 0, -1] + 2 # final chunk end flag accounting for partial chunk
  if parent: flags[:] = 4 # parent flag
  if root: flags[end_idx, 0, 0] = flags[end_idx, 0, 0] + 8 # root flag
  return flags.contiguous()

def compress(data, compressor, count, end_block_len, n_end_blocks, root, parent) -> Tensor:
  IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
  iv = Tensor(IV, dtype=dtypes.uint32).reshape(1, 8, 1).expand(16, 8, data.shape[2]).contiguous()
  states = create_state(iv, count, end_block_len, n_end_blocks, create_flags(data, n_end_blocks, root, parent))
  return compressor(states, data, iv, n_end_blocks)

def blake3(tensor: Tensor) -> str:
  """Hash a Tensor in parallel using the BLAKE3 hashing algorithm."""
  data, n_end_blocks, end_block_len = tensor_to_blake_data(tensor)
  chain_vals = compress(data, compress_chunks, 0, end_block_len, n_end_blocks, data.shape[0] == 1, False)
  tree_levels = math.ceil(math.log2(max(chain_vals.shape[0], 1)))
  # TODO:
  tree_compressor = lambda states, data, iv, _: compress_blocks(states[:, -1], data, iv[:, 0])
  for i in range(tree_levels): # tree-hash chain value pairs ~halving them in each step
    chain_vals, leftover_chain_val = pairwise_concatenate(chain_vals)
    chain_vals = compress(chain_vals, tree_compressor, None, None, None, i == tree_levels - 1, True)
    if leftover_chain_val is not None: chain_vals = chain_vals.cat(leftover_chain_val, dim=0)
  return chain_vals[0].flatten().numpy().tobytes()[:32].hex() # TODO: bitcast to uint8 and slice

if __name__ == "__main__":
  from tinygrad import Device
  print(f"os.environ['DEBUG'] = {os.environ.get('DEBUG')}")
  print(f"Device.DEFAULT = {Device.DEFAULT}")
  kilobyte = 1024
  megabyte = 1024 * kilobyte
  gigabyte = 1024 * megabyte
  t = Tensor.full((kilobyte + 200), fill_value=222, dtype=dtypes.uint8)
  print(f"Input tensor size: {t.shape[0] * t.dtype.itemsize / megabyte:.5f}MB")
  import time
  start = time.monotonic()
  result = blake3(t)
  end = time.monotonic()
  print(f"Hash: {result}")
  print(f"Time taken: {end - start:.3f} seconds")


