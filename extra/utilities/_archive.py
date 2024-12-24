from math import ceil, log2
from typing import Tuple
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import ceildiv

class BLAKE3:
  """BLAKE3 hashing algorithm. Paper: https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf."""

  PAD, DEFAULT_LEN, PERMUTATIONS = 66, 65, Tensor([2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8], dtype=dtypes.uint32)

  def __init__(self): self.compress_blocks_jit = TinyJit(self.compress_blocks)

  def compress_blocks(self, states: Tensor, data: Tensor, chain_vals: Tensor) -> Tensor:
    def rotr(x: Tensor, n: int) -> Tensor: return (x << (32 - n)) | (x >> n)
    for i in range(7):
      for j, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
        mx, my = data[j * 2], data[j * 2 + 1]
        for m in (mx, my):
          states[a] = (states[a] + states[b] + m).cast(dtypes.uint32)
          states[d] = rotr(states[d] ^ states[a], 16 if m is mx else 8).cast(dtypes.uint32)
          states[c] = (states[c] + states[d]).cast(dtypes.uint32)
          states[b] = rotr(states[b] ^ states[c], 12 if m is mx else 7).cast(dtypes.uint32)
      if i < 6: data = data[self.PERMUTATIONS]
    return (states[:8] ^ states[8:]).cat(chain_vals[:8] ^ states[8:])

  @TinyJit
  def init_states(self, data: Tensor, info: Tensor, flags: Tensor) -> Tuple[Tensor, Tensor]:
    IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)
    chain_vals = IV.reshape(1, 8, 1).expand(16, 8, info.shape[-1]).contiguous()
    counts = Tensor.arange(0, data.shape[-1], dtype=dtypes.uint32).reshape(-1, 1).expand(-1, 16).reshape(-1, 16, 1).permute(1, 2, 0)
    counts = counts.cat(Tensor.zeros(chain_vals.shape[0], 1, chain_vals.shape[-1], dtype=dtypes.uint32), dim=1)
    lengths = (info == self.DEFAULT_LEN).where(64, info)
    states = chain_vals.cat(chain_vals[:, :4], counts, lengths, flags, dim=1)# * (info < self.PAD)
    return states.realize(), chain_vals.realize()

  @TinyJit
  def finalize_states(self, states: Tensor, info: Tensor) -> Tensor:
    states = states * (info < self.PAD)
    end_block = (states * (info < self.DEFAULT_LEN)).sum(0)
    return (states[-1, :] | end_block)[:8].realize()
  
  def init_chain_vals(self, data: Tensor, info: Tensor) -> Tuple[Tensor, Tensor]:
    flags = Tensor.zeros((16, 1, info.shape[-1]), dtype=dtypes.uint32).contiguous()
    flags[-1, 0] = (flags[-1, 0] + 2) # chunk end flag
    flags = ((flags != 2) * (info < self.DEFAULT_LEN)).where(flags + 2, flags) # chunk end flag for partial final chunk
    flags[0, :] = flags[0, :] + 1 # chunk start flag
    flags = (((info < self.PAD).sum() <= 16) * (info < self.DEFAULT_LEN)).where(flags + 8, flags) # root flag
    states, chain_vals = self.init_states(data, info, flags)
    for i in range(16):
      next_state = states[i] if i == 0 else states[i-1, :8].cat(states[i, 8:])
      states[i] = self.compress_blocks_jit(next_state.contiguous(), data[i].contiguous(), chain_vals[i].contiguous())
    return self.finalize_states(states, info)

  @TinyJit
  def tree_step(self, chain_vals: Tensor, final_step: Tensor) -> Tensor:
    stacked = chain_vals.transpose().reshape(-1, 16).transpose().reshape(2, 8, -1)
    stacked_mask = stacked.any(1)
    pair_mask, remainder_mask = (stacked_mask[0] * stacked_mask[1]), (stacked_mask[0] ^ stacked_mask[1])
    paired, remainder = (stacked * pair_mask).reshape(16, -1), (stacked * remainder_mask).reshape(16, -1)[:8]
    flags = Tensor.full((1, paired.shape[-1]), 4, dtype=dtypes.uint32)
    flags = final_step.where(12, flags)
    IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)
    iv = IV.reshape(8, 1).expand(8, paired.shape[-1])
    counts = Tensor.zeros((2, paired.shape[-1]), dtype=dtypes.uint32)
    lengths = Tensor.full((1, paired.shape[-1]), 64, dtype=dtypes.uint32)
    states = iv.cat(iv[:4], counts, lengths, flags, dim=0)
    chain_vals = ((self.compress_blocks(states, paired, iv) * pair_mask)[:8] + remainder).realize()
    chain_vals = chain_vals.pad((None, (0, chain_vals.shape[1])))
    return chain_vals.realize()

  def tree_hash(self, chain_vals: Tensor, n_tree_steps: int) -> Tensor:
    print(f"----- tree_hash -----")
    for _ in range(n_tree_steps):
      final_step = chain_vals[0, :3].prod().cast(dtypes.bool).neg()
      print(f"tree_step {_} final_step: {final_step.tolist()}")
      chain_vals = self.tree_step(chain_vals.contiguous(), final_step)
      print(f"step {_}")
    print(f"----- tree_hash done -----")
    return chain_vals.realize()

  def tensor_to_blake_input(self, tensor: Tensor, padded_input_size: int) -> Tuple[Tensor, Tensor, int]:
    assert padded_input_size % 1024 == 0 and padded_input_size & (padded_input_size - 1) == 0, "padded_input_size must be a power of two divisible by 1024"
    blake_input = tensor.flatten().pad((0, (padded_input_size // tensor.element_size()) - tensor.shape[0]))
    blake_input = blake_input.bitcast(dtypes.uint32).reshape(-1, 16, 16).permute(1, 2, 0).contiguous()
    final_chunk_len = 0 if tensor.nbytes() == 0 else (tensor.nbytes() % 1024 or 1024)
    n_end_blocks = ceildiv(final_chunk_len, 64) or 1
    n_chunks = max(1, ceildiv(tensor.nbytes(), 1024))
    info = Tensor.full((16, 1, blake_input.shape[-1]), fill_value=self.DEFAULT_LEN, dtype=dtypes.uint32).contiguous()
    info[n_end_blocks - 1, :, n_chunks - 1] = 0 if tensor.nbytes() == 0 else (tensor.nbytes() % 64) or 64
    info[n_end_blocks:, :, n_chunks - 1:] = info[:, :, n_chunks:] = self.PAD
    return blake_input, info, ceil(log2(max(n_chunks, 1)))

  def hash(self, tensor: Tensor, padded_input_size: int = 1024**2 * 512) -> str:
    print(f"----- init hash -----")
    data, info, n_tree_steps = self.tensor_to_blake_input(tensor, padded_input_size)
    # print(f"init hash output: {data[:, :, :3].numpy()}")
    print(f"----- init hash done -----")
    chain_vals = self.init_chain_vals(data, info)
    chain_vals = self.tree_hash(chain_vals, n_tree_steps) if n_tree_steps > 0 else chain_vals
    return chain_vals[:, 0].flatten().bitcast(dtypes.uint8).data().tobytes().hex()