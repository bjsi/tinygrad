import math
from typing import Optional, Tuple
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

def mix(states: Tensor, chunks: Tensor) -> Tensor:
  def rotr(x: Tensor, n: int) -> Tensor: return ((x << (32 - n)) | (x >> n))
  for i, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
    mx, my = chunks[:, i * 2], chunks[:, i * 2 + 1]
    for m in (mx, my):
      states[:, a] = states[:, a] + states[:, b] + m
      states[:, d] = rotr(states[:, d] ^ states[:, a], 16 if m is mx else 8)
      states[:, c] = states[:, c] + states[:, d]
      states[:, b] = rotr(states[:, b] ^ states[:, c], 12 if m is mx else 7)

def compress_chunks(states: Tensor, chunks: Tensor, chain_vals: Tensor, n_end_blocks: int):
  for i in range(chunks.shape[1]): # parallel over chunks, sequential over blocks
    compressed = compress_blocks(states[:, i].contiguous(), chunks[:, i].contiguous(), chain_vals[:, i].contiguous())
    if i < chunks.shape[1] - 1: states[:, i + 1, :8] = compressed[:, :8] # propagate chain vals to the next block
  end_idx = n_end_blocks - 1 if chunks.shape[0] == 1 else n_end_blocks # handle partial final chunk
  return compressed[:, :8] if n_end_blocks == 16 else compressed[:-1, :8].cat(states[-1:, end_idx, :8])

def compress_blocks(states: Tensor, chunks: Tensor, chain_vals: Tensor) -> Tensor:
  for _ in range(6):
    mix(states, chunks)
    chunks.replace(chunks[:, [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]])
  mix(states, chunks)
  states[:, :8] = states[:, :8] ^ states[:, 8:]
  states[:, 8:] = chain_vals[:, :8] ^ states[:, 8:]
  return states

def bytes_to_chunks(text_bytes: bytes) -> Tuple[Tensor, int, int]:
  chunks_list = []
  for chunk_idx in range(0, max(len(text_bytes), 1), 1024):
    chunk = text_bytes[chunk_idx: chunk_idx + 1024].ljust(1024, b"\x00")
    for block_idx in range(16):
      blocks = chunk[block_idx * 64: (block_idx * 64) + 64]
      chunks_list.append([int.from_bytes(blocks[i: i + 4], "little") for i in range(0, len(blocks), 4)])
  chunks = Tensor(chunks_list, dtype=dtypes.uint32).reshape(max((len(text_bytes) + 1024 - 1) // 1024, 1), 16, 16)
  final_chunk_len = len(text_bytes) - ((chunks.shape[0] - 1) * 1024)
  n_end_blocks = max(1, (final_chunk_len // 64) + (1 if final_chunk_len % 64 else 0))
  end_block_len = 64 if len(text_bytes) % 64 == 0 and len(text_bytes) else len(text_bytes) % 64
  return chunks.contiguous(), n_end_blocks, end_block_len  # chunks is [chunks, blocks, words]

def pairwise_concatenate(chain_vals: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
  leftover_chunk = chain_vals[-1:] if chain_vals.shape[0] % 2 else None
  return chain_vals[:-1 if leftover_chunk is not None else None].reshape(-1, 16), leftover_chunk

def create_state(iv: Tensor, count: Optional[int], end_block_len: Optional[int], n_end_blocks: Optional[int], flags: Tensor) -> Tensor:
  n_chunks, n_blocks = iv.shape[0], iv.shape[1]
  if count is not None:
    counts = Tensor.arange(count, count + n_chunks, dtype=dtypes.uint32)
    counts = counts.reshape(n_chunks, 1).expand(n_chunks, n_blocks).reshape(n_chunks, n_blocks, 1)
    counts = counts.cat(Tensor.zeros(n_chunks, n_blocks, 1, dtype=dtypes.uint32), dim=-1)
  else:
    counts = Tensor.zeros(n_chunks, n_blocks, 2, dtype=dtypes.uint32)
  lengths = Tensor.full((n_chunks, n_blocks, 1), fill_value=64, dtype=dtypes.uint32).contiguous()
  if end_block_len is not None: lengths[-1, n_end_blocks - 1] = end_block_len
  return iv.cat(iv[:, :, :4], counts, lengths, flags, dim=-1).contiguous()

def create_flags(chunks: Tensor, n_end_blocks: Optional[int], root: bool, parent: bool) -> Tensor:
  end_idx = n_end_blocks - 1 if n_end_blocks is not None else -1
  flags = Tensor.zeros((chunks.shape[0], chunks.shape[1], 1), dtype=dtypes.uint32).contiguous()
  flags[:, 0] = flags[:, 0] + 1 # chunk start flag
  flags[:-1, -1] = flags[:-1, -1] + 2 # chunk end flag
  flags[-1, end_idx:] = flags[-1, end_idx:] + 2 # chunk end flag
  if parent: flags[:, :, :] = 4 # parent flag
  if root: flags[0, end_idx] = flags[0, end_idx] + 8 # root flag
  return flags

def compress(data, compressor, count, end_block_len, n_end_blocks, root, parent) -> Tensor:
  IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
  iv = Tensor(IV, dtype=dtypes.uint32).expand(data.shape[0], data.shape[1], -1).contiguous()
  states = create_state(iv, count, end_block_len, n_end_blocks, create_flags(data, n_end_blocks, root, parent))
  return compressor(states, data, iv, n_end_blocks)[:, :8]

def blake3(data: Tensor) -> str:
  """Hash a Tensor in parallel using the BLAKE3 hashing algorithm."""
  chunks, n_end_blocks, end_block_len = bytes_to_chunks(data.flatten().numpy().tobytes())
  chain_vals = compress(chunks, compress_chunks, 0, end_block_len, n_end_blocks, chunks.shape[0] == 1, False)
  tree_levels = math.ceil(math.log2(max(chain_vals.shape[0], 1)))
  tree_compressor = lambda states, data, iv, _: compress_blocks(states[:, -1].contiguous(), data, iv[:, 0])
  for i in range(tree_levels): # tree-hash chain value pairs ~halving them in each step
    chain_vals, leftover_chain_val = pairwise_concatenate(chain_vals)
    chain_vals = compress(chain_vals, tree_compressor, None, None, None, i == tree_levels - 1, True)
    if leftover_chain_val is not None: chain_vals = chain_vals.cat(leftover_chain_val, dim=0)
  return chain_vals[0].flatten().numpy().tobytes()[:32].hex()
