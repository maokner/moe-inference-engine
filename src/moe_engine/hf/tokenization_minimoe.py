"""A local, exact wrapper around miniMoE's tiktoken GPT-2 tokenizer."""

from __future__ import annotations

from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from transformers import PreTrainedTokenizer


class MiniMoETokenizer(PreTrainedTokenizer):
    """Slow tokenizer that preserves tiktoken IDs and decoding byte-for-byte."""

    vocab_files_names = {"vocab_file": "minimoe.tiktoken"}
    model_input_names = ["input_ids", "attention_mask"]
    _PATTERN = (
        "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++| "
        "?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s"
    )

    def __init__(self, vocab_file: str, **kwargs) -> None:
        self.vocab_file = str(vocab_file)
        mergeable_ranks = load_tiktoken_bpe(self.vocab_file)
        self._encoding = tiktoken.Encoding(
            name="minimoe-gpt2",
            pat_str=self._PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens={"<|endoftext|>": 50256},
        )
        kwargs.setdefault("bos_token", "<|endoftext|>")
        kwargs.setdefault("eos_token", "<|endoftext|>")
        kwargs.setdefault("unk_token", "<|endoftext|>")
        kwargs.setdefault("pad_token", "<|endoftext|>")
        kwargs.setdefault("model_max_length", 1024)
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        # The tokenizer remains GPT-2 exactly. The model pads its separate
        # embedding/output vocabulary from 50,257 to 50,304 rows.
        return self._encoding.n_vocab

    def get_vocab(self) -> dict[str, int]:
        return {self._convert_id_to_token(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        del kwargs
        return [str(token_id) for token_id in self._encoding.encode_ordinary(text)]

    def _convert_token_to_id(self, token: str) -> int:
        if token == "<|endoftext|>":
            return 50256
        try:
            return int(token)
        except ValueError:
            return 50256

    def _convert_id_to_token(self, index: int) -> str:
        if index == 50256:
            return "<|endoftext|>"
        return str(index)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        token_ids = [self._convert_token_to_id(token) for token in tokens]
        return self._encoding.decode(token_ids, errors="replace")

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        destination = Path(save_directory) / f"{prefix}minimoe.tiktoken"
        if Path(self.vocab_file).resolve() != destination.resolve():
            destination.write_bytes(Path(self.vocab_file).read_bytes())
        return (str(destination),)
