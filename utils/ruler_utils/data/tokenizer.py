def select_tokenizer(tokenizer_type, tokenizer_path):
    if tokenizer_type != "hf":
        raise ValueError(f"Only the Hugging Face tokenizer is supported, got {tokenizer_type!r}")
    return HFTokenizer(model_path=tokenizer_path)


class HFTokenizer:
    """Tokenizer backed by the model's Hugging Face tokenizer files."""

    def __init__(self, model_path) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def text_to_tokens(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    def tokens_to_text(self, tokens: list[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)
