"""Human-readable labels for model tokens."""

from __future__ import annotations

from tiny_llm_lab.tokenizer import Tokenizer


def _visible_text(text: str) -> str:
    replacements = {" ": "\u2420", "\n": "\u21b5", "\t": "\u21e5", "\r": "\\r"}
    rendered: list[str] = []
    for character in text:
        if character in replacements:
            rendered.append(replacements[character])
        elif character.isprintable():
            rendered.append(character)
        elif ord(character) <= 0xFF:
            rendered.append(f"\\x{ord(character):02x}")
        else:
            rendered.append(f"\\u{ord(character):04x}")
    return "".join(rendered)


def format_token(tokenizer: Tokenizer, token_id: int) -> str:
    """Return a safe label for a token, including standalone byte pieces."""
    try:
        return _visible_text(tokenizer.decode([token_id]))
    except ValueError:
        vocabulary = getattr(tokenizer, "vocabulary", None)
        if isinstance(vocabulary, tuple) and isinstance(vocabulary[token_id], bytes):
            return f"bytes: 0x{vocabulary[token_id].hex()}"
        raise
