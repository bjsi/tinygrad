import math
from typing import Optional, Tuple
import unittest
import numpy as np
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

MD_LEN = 32
HEIGHT, WIDTH = 4, 4
KEY_LEN = 32
BLOCK_BYTES = 64
CHUNK_BYTES = 1024
# flags
CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
PARENT = 1 << 2
ROOT = 1 << 3
IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)
MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]


def rotr(x: int, n: int) -> int:
  return (x << (32 - n)) | (x >> n)


def mix(states: Tensor, a: int, b: int, c: int, d: int, mx: Tensor, my: Tensor) -> Tensor:
  states[:, a] = states[:, a] + (states[:, b] + mx)
  states[:, d] = rotr(states[:, d] ^ states[:, a], 16)
  states[:, c] = states[:, c] + states[:, d]
  states[:, b] = rotr(states[:, b] ^ states[:, c], 12)
  states[:, a] = states[:, a] + (states[:, b] + my)
  states[:, d] = rotr(states[:, d] ^ states[:, a], 8)
  states[:, c] = states[:, c] + states[:, d]
  states[:, b] = rotr(states[:, b] ^ states[:, c], 7)


def round(states: Tensor, chunks: Tensor) -> Tensor:
  # mix columns
  mix(states, 0, 4, 8, 12, chunks[:, 0], chunks[:, 1])
  mix(states, 1, 5, 9, 13, chunks[:, 2], chunks[:, 3])
  mix(states, 2, 6, 10, 14, chunks[:, 4], chunks[:, 5])
  mix(states, 3, 7, 11, 15, chunks[:, 6], chunks[:, 7])
  # mix diagonals
  mix(states, 0, 5, 10, 15, chunks[:, 8], chunks[:, 9])
  mix(states, 1, 6, 11, 12, chunks[:, 10], chunks[:, 11])
  mix(states, 2, 7, 8, 13, chunks[:, 12], chunks[:, 13])
  mix(states, 3, 4, 9, 14, chunks[:, 14], chunks[:, 15])


def permute(chunks: Tensor) -> Tensor:
  original = chunks.clone()
  for i in range(16):
    chunks[:, i] = original[:, MSG_PERMUTATION[i]]
  return chunks


def round_and_permute(states: Tensor, chunks: Tensor):
  for _ in range(6):
    round(states, chunks)
    permute(chunks)
  round(states, chunks)

def compress_chunks(states: Tensor, chunks: Tensor, chain_vals: Tensor, n_end_blocks: Optional[int]):
  """
  Chunk compression can be done in parallel because each chunk is independent.
  Block compression is sequential because each chunk depends on the chaining value of the previous block.
  states: [n_chunks, n_blocks, state_size]
  chunks: [n_chunks, n_blocks, n_words]
  chain_vals: [n_chunks, n_blocks, 8]
  returns: [n_chunks, 8] chunk chaining values
  """
  for i in range(states.shape[1]):
    compressed_states = compress_block(states[:, i].contiguous(), chunks[:, i].contiguous(), chain_vals[:, i].contiguous())
    compressed_chaining_vals = compressed_states[:, :8]
    if i < states.shape[1] - 1:
      chain_vals[:, i + 1] = compressed_chaining_vals
      states[:, i + 1, :8] = compressed_chaining_vals
  # each chunk's chaining value is the chaining value of its final block
  end_idx = n_end_blocks if n_end_blocks is not None else -1
  return chain_vals[:-1, -1].cat(chain_vals[-1:, end_idx])

def compress_block(
    states: Tensor,
    chunks: Tensor,
    chain_vals: Tensor,
) -> Tensor:
  """
  states: [n_chunks, state_size]
  chunks: [n_chunks, n_words]
  chain_vals: [n_chunks, 8]
  """
  round_and_permute(states, chunks)
  for i in range(8):
    states[:, i] = states[:, i] ^ states[:, i + 8]
    states[:, i + 8] = states[:, i + 8] ^ chain_vals[:, i]
  return states


def bytes_to_blocks(text_bytes: bytes) -> Tuple[Tensor, int, int]:
  """
  Each chunk is 1024 bytes made up of 16 blocks.
  Each block contains 16 32-bit words.
  returns: ([n_chunks, n_blocks, n_words], n_end_blocks, end_block_len)
  """
  n_bytes = len(text_bytes)
  n_chunks = max((n_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES, 1)
  chunks = Tensor.zeros(n_chunks, 16, 16, dtype=dtypes.uint32).contiguous()
  unpadded_len = 0
  for i in range(0, max(len(text_bytes), 1), CHUNK_BYTES):
    chunk = text_bytes[i:i + CHUNK_BYTES]
    unpadded_len = len(chunk)
    chunk = chunk.ljust(CHUNK_BYTES, b"\0")
    for j in range(16):
      block_start = j * BLOCK_BYTES
      b = chunk[block_start:block_start + BLOCK_BYTES]
      block_words = [int.from_bytes(b[i: i + 4], "little") for i in range(0, len(b), 4)]
      chunks[i // CHUNK_BYTES, j] = Tensor(block_words, dtype=dtypes.uint32)
  n_end_blocks = max(1, (unpadded_len // BLOCK_BYTES) + (1 if unpadded_len % BLOCK_BYTES else 0))
  end_block_len = BLOCK_BYTES if unpadded_len % BLOCK_BYTES == 0 and unpadded_len else unpadded_len % BLOCK_BYTES
  return chunks, n_end_blocks, end_block_len


def pair_chaining_values(chain_vals: Tensor) -> Tensor:
  """
  Pairwise concatenate chaining values to create inputs for the parent level of the hash tree.
  """
  n_chunks = chain_vals.shape[0]
  assert chain_vals.shape == (n_chunks, 8)
  return chain_vals.reshape(-1 // 2, 2, 4).flatten(1)


def create_state(chain_vals: Tensor, iv: Tensor, counter: int, last_len: Optional[int], n_end_blocks: Optional[int], flags: Tensor) -> Tensor:
  """
  returns: [n_chunks, n_blocks, state_size]
  """
  n_chunks, n_blocks, _ = chain_vals.shape
  counts = Tensor.arange(counter, counter + n_chunks, dtype=dtypes.uint32)
  counts = counts.cat(Tensor.zeros_like(counts, dtype=dtypes.uint32))
  counts = counts.expand(n_chunks, n_blocks, -1)
  lengths = Tensor.full((n_chunks, n_blocks, 1), fill_value=BLOCK_BYTES, dtype=dtypes.uint32).contiguous()
  if last_len is not None:
    lengths[:, n_end_blocks - 1] = last_len
    lengths[:, n_end_blocks:] = 0
  states = chain_vals.cat(iv[:, :, :4], counts, lengths, flags, dim=-1)
  return states

def create_flags(n_chunks: int, n_blocks: int, n_end_blocks: Optional[int], parents: bool, root: bool) -> Tensor:
  flags = Tensor.zeros((n_chunks, n_blocks, 1), dtype=dtypes.uint32).contiguous()
  flags[:, 0] = flags[:, 0] + CHUNK_START
  end_idx = n_end_blocks - 1 if n_end_blocks is not None else -1
  flags[:, end_idx] = flags[:, end_idx] + CHUNK_END
  if parents:
    flags = flags + PARENT
  if root:
    flags[:, end_idx] = flags[:, end_idx] + ROOT
  return flags


def tiny_blake3(text: str) -> str:
  text_bytes = text.encode("utf-8") if text else b""
  chunks, n_end_blocks, end_block_len = bytes_to_blocks(text_bytes)
  iv = IV.expand(chunks.shape[0], chunks.shape[1], -1).contiguous()
  counter = 0
  # tree hash
  tree_levels = math.ceil(math.log2(max(chunks.shape[0], 1)))
  for i in range(tree_levels):
    chain_vals = iv if i == 0 else None
    flags = create_flags(chunks.shape[0], chunks.shape[1], n_end_blocks, i > 0)
    states = create_state(chain_vals, iv, counter, end_block_len, n_end_blocks, flags)
    states = compress_chunks(states, chunks, chain_vals, n_end_blocks)
    counter += chunks.shape[0]
  # root hash
  flags = create_flags(chunks.shape[0], chunks.shape[1], n_end_blocks, counter != 0, True)
  states = create_state(iv, iv, counter, end_block_len, n_end_blocks, flags)
  chain_vals = compress_chunks(states, chunks, iv, n_end_blocks)
  hash = chain_vals[0].flatten().numpy().tobytes()[:MD_LEN].hex()
  return hash


class TestBLAKE3(unittest.TestCase):
  def test_bytes_to_blocks(self):
    # empty
    text = b""
    actual, n_end_blocks, end_block_len = bytes_to_blocks(text)
    self.assertEqual(actual.shape, (1, 16, 16))
    self.assertEqual(n_end_blocks, 1)
    self.assertEqual(end_block_len, 0)
    np.testing.assert_equal(actual.numpy(), np.zeros((1, 16, 16)))
    # single byte
    text = b"a"
    actual, n_end_blocks, end_block_len = bytes_to_blocks(text)
    self.assertEqual(n_end_blocks, 1)
    self.assertEqual(end_block_len, 1)
    np.testing.assert_equal(actual[0, 0, 0].numpy(), 97)
    # seven bytes
    text = b"abcdefg"
    actual, n_end_blocks, end_block_len = bytes_to_blocks(text)
    self.assertEqual(n_end_blocks, 1)
    self.assertEqual(end_block_len, 7)
    expected = [1684234849, 6776421, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_equal(actual[0, 0].numpy(), expected)
    # two chunks
    text = (b"a" * CHUNK_BYTES) + (b"b" * CHUNK_BYTES)
    actual, n_end_blocks, end_block_len = bytes_to_blocks(text)
    self.assertEqual(n_end_blocks, 16)
    self.assertEqual(end_block_len, 64)
    self.assertEqual(actual.shape, (2, 16, 16))
    np.testing.assert_equal(actual[0, 0, 0].numpy(), 1633771873)
    np.testing.assert_equal(actual[1, 0, 0].numpy(), 1650614882)
    # unicode
    text = "🤖 你好"
    actual, n_end_blocks, end_block_len = bytes_to_blocks(text.encode("utf-8"))
    self.assertEqual(n_end_blocks, 1)
    self.assertEqual(end_block_len, 11)
    self.assertEqual(actual.shape, (1, 16, 16))
    expected = [2527371248, 2696799264, 12428773, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_equal(actual[0, 0].numpy(), expected)

  def test_empty(self):
    actual = tiny_blake3("")
    expected = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    self.assertEqual(actual, expected)

  def test_single_char(self):
    actual = tiny_blake3("a")
    expected = "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"
    self.assertEqual(actual, expected)

  def test_short(self):
    actual = tiny_blake3("abcdefg")
    expected = "e2d18d70db12705e1845faf500de1198a5ba1483729d97936f1d2b760968312e"
    self.assertEqual(actual, expected)

  def test_block(self):
    text = "abcd" * (64 // 4)
    actual = tiny_blake3(text)
    expected = '0ef2431cde7c3268b417ea0e8c692dafa8211df7d59f09fdb23df4d73a3bd43d'
    self.assertEqual(actual, expected)

  def test_block_plus_one(self):
    text = "a" * (BLOCK_BYTES + 1)
    actual = tiny_blake3(text)
    expected = 'f345679d9055e53939e92c04ff4f6c9d824b849810d4b598f54baa23336cde99'
    self.assertEqual(actual, expected)

  def test_multiple_blocks(self):
    text = ("a" * BLOCK_BYTES) + ("b" * BLOCK_BYTES)
    actual = tiny_blake3(text)
    expected = 'f27ee0ad41ba8d44a592347ad98c260260d36a59aae97b8e8abc51a3f087bff7'
    self.assertEqual(actual, expected)
    text = ("a" * BLOCK_BYTES) + ("b" * BLOCK_BYTES) + ("c" * BLOCK_BYTES) + ("d" * BLOCK_BYTES)
    actual = tiny_blake3(text)
    expected = 'a9089941f4dc9da1f32e5b037cfe53b2b07feb7ab2ef562444af540333a9e605'
    self.assertEqual(actual, expected)


if __name__ == "__main__":
  unittest.main()
