from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

@dataclass
class CommonVoiceRow:
    path: str
    sentence: str
    raw: Dict[str, str]

    @property
    def filename_stem(self) -> str:
        return Path(self.path).stem

def load_commonvoice_test_rows(test_tsv_path: Path) -> List[CommonVoiceRow]:
    rows: List[CommonVoiceRow] = []
    test_tsv_path = Path(test_tsv_path)

    with test_tsv_path.open("r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
        if not header_line:
            raise ValueError("test.tsv is empty")
        header = header_line.split("\t")

        if "path" not in header or "sentence" not in header:
            raise ValueError("test.tsv must contain 'path' and 'sentence' columns")

        for line_num, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t")

            # If column count is mismatched, apply a tolerant fallback.
            if len(parts) < len(header):
                # Corrupted/incomplete row: skip (could also be logged).
                continue
            elif len(parts) > len(header):
                # Assume extra tabs belong to the last column.
                parts = parts[:len(header)-1] + ["\t".join(parts[len(header)-1:])]

            rec = dict(zip(header, parts))
            path_val = rec.get("path", "").strip()
            sentence_val = rec.get("sentence", "").strip()

            if not path_val:
                continue

            rows.append(CommonVoiceRow(path=path_val, sentence=sentence_val, raw=rec))

    return rows
