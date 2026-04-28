"""测试共用的辅助对象。

这里放一个足够稳定、可逆的假 tokenizer，避免单元测试依赖真实模型权重或
transformers tokenizer 下载。它不追求“像真实 tokenizer 一样复杂”，只要求：

- `encode -> decode -> encode` 基本稳定；
- 不同文本通常能得到不同 token 序列；
- 换行和空格边界比较可控，方便测试 prefix 对齐。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import sys


# 让 pytest 从仓库根目录解析 `src.*` 导入，避免每个测试文件单独改 sys.path。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DummyTokenizer:
    """一个简单、稳定、可逆的测试 tokenizer。"""

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._next_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        pieces = self._split_text(text)
        return [self._get_or_create_id(piece) for piece in pieces]

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = False,
    ) -> str:
        del skip_special_tokens
        pieces = [self._id_to_token[token_id] for token_id in token_ids]
        return "".join(pieces)

    def _get_or_create_id(self, piece: str) -> int:
        token_id = self._token_to_id.get(piece)
        if token_id is not None:
            return token_id

        token_id = self._next_id
        self._next_id += 1
        self._token_to_id[piece] = token_id
        self._id_to_token[token_id] = piece
        return token_id

    @staticmethod
    def _split_text(text: str) -> list[str]:
        pieces: list[str] = []
        current = []

        def flush_current() -> None:
            if current:
                pieces.append("".join(current))
                current.clear()

        for char in text:
            if char == "\n":
                flush_current()
                pieces.append("\n")
            elif char.isspace():
                flush_current()
                pieces.append(" ")
            elif char.isalnum() or char == "_":
                current.append(char)
            else:
                flush_current()
                pieces.append(char)

        flush_current()
        return pieces
