import re
import unicodedata
from dataclasses import dataclass
from typing import Mapping

try:
    from num2words import num2words as _num2words
except Exception:
    _num2words = None

# Hebrew diacritics and cantillation marks
HEBREW_DIACRITICS_RE = re.compile(r"[\u0591-\u05C7]")
MULTISPACE_RE = re.compile(r"\s+")
INTEGER_TOKEN_RE = re.compile(r"^[+-]?\d+$")
PREFIXED_INTEGER_RE = re.compile(r"(?<!\w)([בלכהמשו]?)(?:[-־])?(\d[\d,_]*)(?=[^\d]|$)")

# Character normalization mapping
CHAR_REPLACEMENTS = {
    "״": '"',
    "“": '"',
    "”": '"',
    "„": '"',
    "׳": "'",
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "־": "-",  # Hebrew maqaf
}

# Punctuation to strip (while keeping letters/digits/spaces)
PUNCT_TO_SPACE_RE = re.compile(r"""[.,!?;:()\[\]{}<>/"\\|*_+=~`@#$%^&]+""")

# Spelling-equivalence map ("not really errors") from frequent Stage C patterns
DEFAULT_EQUIV_MAP: dict[str, str] = {
    "הייתה": "היתה",
    "איתי": "אתי",
    "מיסים": "מסים",
    "הכל": "הכול",
    "מיד": "מייד",
    "ואילו": "ואלו",
    "שמיים": "שמים",
}


@dataclass(frozen=True)
class NormalizationConfig:
    remove_diacritics: bool = True
    hyphen_to_space: bool = True
    remove_punctuation: bool = True
    convert_numbers_to_words: bool = False
    word_equivalents: Mapping[str, str] | None = None


STAGE_C_STEP1_PUNCT = NormalizationConfig(
    remove_diacritics=False,
    hyphen_to_space=False,
    remove_punctuation=True,
    word_equivalents=None,
)

STAGE_C_STEP2_PUNCT_PLUS_HYPHEN = NormalizationConfig(
    remove_diacritics=False,
    hyphen_to_space=True,
    remove_punctuation=True,
    word_equivalents=None,
)

STAGE_C_STEP3_PLUS_DIACRITICS = NormalizationConfig(
    remove_diacritics=True,
    hyphen_to_space=True,
    remove_punctuation=True,
    word_equivalents=None,
)

STAGE_C_COMPLETE = NormalizationConfig(
    remove_diacritics=True,
    hyphen_to_space=True,
    remove_punctuation=True,
    convert_numbers_to_words=False,
    word_equivalents=DEFAULT_EQUIV_MAP,
)

STAGE_C_COMPLETE_PLUS_NUMBERS = NormalizationConfig(
    remove_diacritics=True,
    hyphen_to_space=True,
    remove_punctuation=True,
    convert_numbers_to_words=True,
    word_equivalents=DEFAULT_EQUIV_MAP,
)


def _apply_char_replacements(text: str) -> str:
    for src, dst in CHAR_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def _convert_number_token(token: str) -> str:
    candidate = token.replace(",", "").replace("_", "")
    if not INTEGER_TOKEN_RE.fullmatch(candidate):
        return token

    if _num2words is None:
        raise RuntimeError(
            "num2words is required for number normalization. Install with: pip install num2words"
        )

    try:
        number_value = int(candidate)
        return str(_num2words(number_value, lang="he"))
    except Exception:
        return token


def _convert_number_match(match: re.Match[str]) -> str:
    prefix = match.group(1) or ""
    numeric_part = match.group(2)
    number_words = _convert_number_token(numeric_part)
    if number_words == numeric_part:
        return match.group(0)

    if prefix:
        parts = number_words.split()
        if not parts:
            return match.group(0)
        parts[0] = prefix + parts[0]
        return " ".join(parts)

    return number_words


def _numbers_to_words_text(text: str) -> str:
    return PREFIXED_INTEGER_RE.sub(_convert_number_match, text)


def normalize_text(text: str, config: NormalizationConfig) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = _apply_char_replacements(text)

    if config.remove_diacritics:
        text = HEBREW_DIACRITICS_RE.sub("", text)

    if config.convert_numbers_to_words:
        text = _numbers_to_words_text(text)
        text = _apply_char_replacements(text)
        if config.remove_diacritics:
            text = HEBREW_DIACRITICS_RE.sub("", text)

    if config.hyphen_to_space:
        text = text.replace("-", " ")

    if config.remove_punctuation:
        text = PUNCT_TO_SPACE_RE.sub(" ", text)

    text = MULTISPACE_RE.sub(" ", text).strip()

    if config.word_equivalents:
        words = text.split()
        words = [config.word_equivalents.get(w, w) for w in words]
        text = " ".join(words)

    return text


def make_normalizer(config: NormalizationConfig):
    def _normalizer(text: str) -> str:
        return normalize_text(text, config=config)

    return _normalizer


def reference_normalize(text: str) -> str:
    return normalize_text(text, config=STAGE_C_COMPLETE)


def transcription_normalize(text: str) -> str:
    return normalize_text(text, config=STAGE_C_COMPLETE)
