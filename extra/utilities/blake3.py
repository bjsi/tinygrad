import math
import os
from typing import List, Optional, Tuple
from tinygrad.dtype import dtypes
from tinygrad.engine import jit
from tinygrad.tensor import Tensor

def rotr(x: Tensor, n: int) -> Tensor: return ((x << (32 - n)) | (x >> n))

def mix_1(word_list: List[Tensor], data: Tensor):
  for i, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
    mx, my = data[i * 2], data[i * 2 + 1]
    for m in (mx, my):
      word_list[a] = (word_list[a] + word_list[b] + m)
      word_list[d] = rotr(word_list[d] ^ word_list[a], 16 if m is mx else 8)
      word_list[c] = (word_list[c] + word_list[d])
      word_list[b] = rotr(word_list[b] ^ word_list[c], 12 if m is mx else 7)
    
def mix_2(states: Tensor, data: Tensor):
  for i, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
    mx, my = data[i * 2], data[i * 2 + 1]
    for m in (mx, my):
      states[a] = (states[a] + states[b] + m)
      states[d] = rotr(states[d] ^ states[a], 16 if m is mx else 8)
      states[c] = (states[c] + states[d])
      states[b] = rotr(states[b] ^ states[c], 12 if m is mx else 7)

@jit
def mix_3(states: Tensor, data: Tensor):
  # round 0
  mx, my = data[0], data[1]
  t0 = states[0] + states[4] + mx
  t12 = rotr(states[12] ^ t0, 16)
  t8 = states[8] + t12
  t4 = rotr(states[4] ^ t8, 12)
  t0 = t0 + t4 + my
  t12 = rotr(t12 ^ t0, 8)
  t8 = t8 + t12
  t4 = rotr(t4 ^ t8, 7)

  # round 1
  mx, my = data[2], data[3]
  t1 = states[1] + states[5] + mx
  t13 = rotr(states[13] ^ t1, 16)
  t9 = states[9] + t13
  t5 = rotr(states[5] ^ t9, 12)
  t1 = t1 + t5 + my
  t13 = rotr(t13 ^ t1, 8)
  t9 = t9 + t13
  t5 = rotr(t5 ^ t9, 7)

  # round 2
  mx, my = data[4], data[5]
  t2 = states[2] + states[6] + mx
  t14 = rotr(states[14] ^ t2, 16)
  t10 = states[10] + t14
  t6 = rotr(states[6] ^ t10, 12)
  t2 = t2 + t6 + my
  t14 = rotr(t14 ^ t2, 8)
  t10 = t10 + t14
  t6 = rotr(t6 ^ t10, 7)

  # round 3
  mx, my = data[6], data[7]
  t3 = states[3] + states[7] + mx
  t15 = rotr(states[15] ^ t3, 16)
  t11 = states[11] + t15
  t7 = rotr(states[7] ^ t11, 12)
  t3 = t3 + t7 + my
  t15 = rotr(t15 ^ t3, 8)
  t11 = t11 + t15
  t7 = rotr(t7 ^ t11, 7)

  # round 4
  mx, my = data[8], data[9]
  u0 = t0 + t5 + mx
  u15 = rotr(t15 ^ u0, 16)
  u10 = t10 + u15
  u5 = rotr(t5 ^ u10, 12)
  u0 = u0 + u5 + my
  u15 = rotr(u15 ^ u0, 8)
  u10 = u10 + u15
  u5 = rotr(u5 ^ u10, 7)

  # round 5
  mx, my = data[10], data[11]
  u1 = t1 + t6 + mx
  u12 = rotr(t12 ^ u1, 16)
  u11 = t11 + u12
  u6 = rotr(t6 ^ u11, 12)
  u1 = u1 + u6 + my
  u12 = rotr(u12 ^ u1, 8)
  u11 = u11 + u12
  u6 = rotr(u6 ^ u11, 7)

  # round 6
  mx, my = data[12], data[13]
  u2 = t2 + t7 + mx
  u13 = rotr(t13 ^ u2, 16)
  u8 = t8 + u13
  u7 = rotr(t7 ^ u8, 12)
  u2 = u2 + u7 + my
  u13 = rotr(u13 ^ u2, 8)
  u8 = u8 + u13
  u7 = rotr(u7 ^ u8, 7)

  # round 7
  mx, my = data[14], data[15]
  u3 = t3 + t4 + mx
  u14 = rotr(t14 ^ u3, 16)
  u9 = t9 + u14
  u4 = rotr(t4 ^ u9, 12)
  u3 = u3 + u4 + my
  u14 = rotr(u14 ^ u3, 8)
  u9 = u9 + u14
  u4 = rotr(u4 ^ u9, 7)

  # Write back final values
  states[0] = u0
  states[1] = u1
  states[2] = u2
  states[3] = u3
  states[4] = u4
  states[5] = u5
  states[6] = u6
  states[7] = u7
  states[8] = u8
  states[9] = u9
  states[10] = u10
  states[11] = u11
  states[12] = u12
  states[13] = u13
  states[14] = u14
  states[15] = u15






def compress_chunks(states: Tensor, data: Tensor, chain_vals: Tensor, n_end_blocks: int):
  """Compute over tensors, storing intermediate tensors in lists."""
  word_list: List[Tensor] = [states[0, i] for i in range(8)] # init with first 8 words of the first block
  end_idx = n_end_blocks - 1 if data.shape[2] == 1 else n_end_blocks # to handle partial final chunk
  for i in range(states.shape[0]): # parallel over words, sequential over blocks
    word_list = word_list[:8] + [states[i, j] for j in range(8, 16)] # propagate chain vals to the next block
    word_list = compress_blocks(word_list, data[i], chain_vals[i])
    if i == end_idx: final_chunk_compressed = word_list # save a reference to the end idx block
  # TODO: across all chunks, until the last 
  for word in word_list:
    x = word
    x.realize()

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


