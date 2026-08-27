import os
from pathlib import Path
from typing import Callable

from huggingface_hub import login


def _read_text_if_present(path: Path) -> str | None:
    try:
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            return text or None
    except OSError:
        return None
    return None


def resolve_hf_token() -> str | None:
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        token = os.environ.get(env_name)
        if token:
            return token.strip()

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    for candidate in (hf_home / "token", Path.home() / ".cache" / "huggingface" / "token"):
        token = _read_text_if_present(candidate)
        if token:
            return token
    return None


def setup_hf_auth(logger: object | None = None, *, prefer_login: bool = False) -> bool:
    token = resolve_hf_token()
    if not token:
        return False

    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        os.environ.setdefault(env_name, token)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    if os.environ.get("SKIP_HF_LOGIN", "0") == "1" and not prefer_login:
        return True

    try:
        login(token=token, add_to_git_credential=False, new_session=False)
        return True
    except Exception as exc:  # pragma: no cover - best-effort auth
        if logger is not None:
            log_fn: Callable[[str, object], None] | None = getattr(logger, "warning", None)
            if callable(log_fn):
                log_fn("Hugging Face login skipped after auth setup error: %s", exc)
        return False
