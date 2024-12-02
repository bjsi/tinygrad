import math
from typing import Callable, List, Optional, Tuple
from tinygrad.dtype import dtypes
from tinygrad.engine import jit
from tinygrad.helpers import ceildiv
from tinygrad.tensor import Tensor

class BLAKE3:
  def __init__(self, std_sizes: Optional[List[int]] = None):
    self.IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)
    self.std_sizes = std_sizes or [1024**3 * 3]
    self.PAD, self.DEFAULT_LEN, self.PERM = 66, 65, Tensor([2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8], dtype=dtypes.uint32)

  @jit.TinyJit
  def mix(self, states: Tensor, chunks: Tensor) -> Tensor:
    def rotr(x: Tensor, n: int) -> Tensor: return ((x << (32 - n)) | (x >> n))
    for i, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
      mx, my = chunks[i * 2], chunks[i * 2 + 1]
      for m in (mx, my):
        states[a] = states[a] + states[b] + m
        states[d] = rotr(states[d] ^ states[a], 16 if m is mx else 8)
        states[c] = states[c] + states[d]
        states[b] = rotr(states[b] ^ states[c], 12 if m is mx else 7)

  def compress_chunks(self, states: Tensor, data: Tensor, chain_vals: Tensor, info: Tensor) -> Tensor:
    for i in range(16): # parallel over chunks, sequential over blocks
      self.compress_blocks(states[i].contiguous(), data[i].contiguous(), chain_vals[i].contiguous())
      if i < data.shape[1] - 1: states[i + 1, :8] = states[i, :8] # propagate chain vals
    states = states * (info < self.PAD) # zero out padding
    end_block = (states * (info < self.DEFAULT_LEN)).sum(0) # pick out the end block
    return (states[-1, :] | end_block)[:8] # combine last block of each chunk with end block

  def compress_blocks(self, states: Tensor, data: Tensor, chain_vals: Tensor) -> Tensor:
    for _ in range(6):
      self.mix(states, data)
      data = data[self.PERM]
    self.mix(states, data)
    states[:8] = states[:8] ^ states[8:]
    states[8:] = chain_vals[:8] ^ states[8:]
    return states

  def tensor_to_blake_data(self, tensor: Tensor) -> Tuple[Tensor, Tensor]:
    size = min(size for size in self.std_sizes if size >= tensor.nbytes()) // tensor.element_size()
    data = tensor.flatten().pad(((0, size - tensor.shape[0],),), value=0)
    data = data.bitcast(dtypes.uint32).reshape(-1, 16, 16).permute(1, 2, 0).contiguous()
    final_chunk_len = 0 if tensor.nbytes() == 0 else (tensor.nbytes() % 1024 or 1024)
    n_end_blocks = ceildiv(final_chunk_len, 64) or 1
    n_chunks = max(1, ceildiv(tensor.nbytes(), 1024))
    info = Tensor.full((16, 1, data.shape[-1]), fill_value=self.DEFAULT_LEN, dtype=dtypes.uint32).contiguous()
    info[n_end_blocks - 1, :, n_chunks - 1] = 0 if tensor.nbytes() == 0 else (tensor.nbytes() % 64) or 64
    info[n_end_blocks:, :, n_chunks - 1:] = info[:, :, n_chunks:] = self.PAD
    n_steps = math.ceil(math.log2(max(n_chunks, 1)))
    return data.contiguous(), info.contiguous(), n_steps

  def pairwise_concat(self, chain_vals: Tensor) -> Tuple[Tensor, Tensor]:
    paired_cvs_with_leftover = chain_vals.permute(1, 0).reshape(-1, 16).transpose()
    paired = chain_vals.any(0).reshape(-1, 2)
    paired_mask = (paired[:, 0] * paired[:, 1])
    leftover_mask = (paired[:, 0] ^ paired[:, 1])
    paired_cvs = (paired_cvs_with_leftover * paired_mask).pad(((0, 0), (0, paired.shape[0])), value=0)
    leftover = (paired_cvs_with_leftover * leftover_mask).sum(1)[:8].reshape(-1, 1)
    return paired_cvs, leftover

  def create_state(self, iv: Tensor, counts: Tensor, info: Tensor, flags: Tensor) -> Tensor:
    counts = counts.cat(Tensor.zeros(iv.shape[0], 1, iv.shape[-1], dtype=dtypes.uint32), dim=1)
    lengths = (info == self.DEFAULT_LEN).where(64, info) # set default lengths
    states = iv.cat(iv[:, :4], counts, lengths, flags, dim=1) # create states
    return states * (info < self.PAD).cast(dtypes.uint32) # zero out padding

  def create_flags(self, info: Tensor, parents: Tensor, final_step: Tensor) -> Tensor:
    flags = Tensor.zeros((16, 1, info.shape[-1]), dtype=dtypes.uint32).contiguous()
    flags[-1, 0] = flags[-1, 0] + 2 # chunk end flag
    flags = (flags + 2 * ((flags != 2) * (info < self.DEFAULT_LEN))) # chunk end flag for partial final chunk
    flags[0] = flags[0] + 1 # chunk start flag
    flags = parents.where(4, flags) # parent flag
    flags = (flags + (8 * (((info < self.PAD).sum() <= 16) * (info < self.DEFAULT_LEN)))) # root flag if <= 1 chunk
    flags = final_step.where(12, flags) # final step flag
    flags = flags * (info < self.PAD)
    return flags.cast(dtypes.uint32)

  def compress(self, data: Tensor, compressor: Callable, counts: Tensor, info: Tensor, parents: Tensor, final_step: Tensor) -> Tensor:
    iv = self.IV.reshape(1, 8, 1).expand(16, 8, info.shape[-1]).contiguous()
    flags = self.create_flags(info, parents, final_step)
    states = self.create_state(iv, counts, info, flags)
    return compressor(states, data, iv, info)

  def compress_tree(self, states, data, iv, _): return self.compress_blocks(states[-1].contiguous(), data, iv[0])

  def _hash(self, data: Tensor, info: Tensor, n_steps: int) -> Tensor:
    parents = Tensor.zeros((16, 1, data.shape[-1]), dtype=dtypes.bool).contiguous()
    final_step = Tensor.zeros((16, 1, data.shape[-1]), dtype=dtypes.bool).contiguous()
    counts = Tensor.arange(0, data.shape[-1], dtype=dtypes.uint32).reshape(-1, 1).expand(-1, 16).reshape(-1, 16, 1).permute(1, 2, 0)
    chain_vals = self.compress(data, self.compress_chunks, counts, info, parents, final_step)
    info = (info < self.DEFAULT_LEN).where(64, info)
    counts = Tensor.zeros((16, 1, data.shape[-1]), dtype=dtypes.uint32)
    parents = Tensor.ones((16, 1, data.shape[-1]), dtype=dtypes.bool)
    final_step = chain_vals.any(0).sum(-1) == 2
    results = Tensor.zeros((n_steps + 1, 8), dtype=dtypes.uint32).contiguous()
    results[0] = chain_vals[:, 0]
    for i in range(n_steps): # tree-hash chain value pairs ~halving them in each step
      chain_vals, leftover_chain_val = self.pairwise_concat(chain_vals)
      valid = chain_vals.any(0)
      chain_vals = self.compress(chain_vals.contiguous(), self.compress_tree, counts, info, parents, final_step)[:8] * valid
      results[i + 1] = chain_vals[:, 0]
      insertion_mask = (valid ^ valid.roll(1, -1))
      insertion_mask[0] = 0
      chain_vals = insertion_mask.where(leftover_chain_val, chain_vals)
      final_step = chain_vals.any(0).sum(-1) == 2
    return results.realize()

  def hash(self, tensor: Tensor) -> str:
    data, info, n_steps = self.tensor_to_blake_data(tensor)
    results = self._hash(data, info, n_steps)
    return results[n_steps].flatten().bitcast(dtypes.uint8).data().tobytes().hex()

if __name__ == "__main__":
  # t = Tensor.full((1024 ** 2) * 500, fill_value=1, dtype=dtypes.uint8)
  # print(BLAKE3().hash(t))
  import time
  import sys
  import random

  arg = sys.argv[1]

  if arg == "warmup":
    # warmup the JIT
    print("\nWarming up...")
    def warmup(size):
      print(f"Warming up {size / 1024 / 1024 :.1f} MB...")
      warmup_data = Tensor.rand(size // 2, dtype=dtypes.float16)
      BLAKE3().hash(warmup_data)
    for size in BLAKE3().std_sizes: warmup(size)
  else:
    def benchmark_size(size_bytes):
      print(f"\nBenchmarking {size_bytes / 1024 / 1024 :.1f} MB...")
      randint = random.randint(0, 255)
      data = Tensor.full(size_bytes // 2, fill_value=randint, dtype=dtypes.float16)
      input_size = data.nbytes()
      # data = data.pad(((0, (1024**3 - data.nbytes()) // data.element_size(),),), value=0)
      padded_size = data.numel() * data.element_size()
      print(f"Padded size: {padded_size / 1024 / 1024 :.1f} MB")

      start = time.time()
      BLAKE3().hash(data)
      end = time.time()

      elapsed = end - start
      throughput = input_size / elapsed / 1e6  # MB/s
      print(f"Time: {elapsed:.2f}s")
      print(f"Throughput: {throughput:.1f} MB/s")

    size_mb = float(sys.argv[1])
    randint = random.randint(0, 1024 * 1024 * 20)
    size = int(size_mb * 1024 * 1024) - randint

    benchmark_size(size)
