from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from tqdm import tqdm

from faster_whisper import WhisperModel

@dataclass
class WhisperHebrewTranscriber:
    model_name_or_path: str
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "he"

    def __post_init__(self):
        self.model = WhisperModel(
            self.model_name_or_path,
            device=self.device,
            compute_type=self.compute_type
        )

    def transcribe_file(self, audio_path: Path) -> str:
        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=5,
            vad_filter=False,   # Can be tuned later if needed.
            condition_on_previous_text=False,
        )
        text = "".join(seg.text for seg in segments).strip()
        return " ".join(text.split())

def transcribe_benchmark_to_tsv(
    rows,
    clips_dir: Path,
    output_tsv: Path,
    transcriber: WhisperHebrewTranscriber,
    resume: bool = True,
):
    """
    Writes TSV columns:
    Filename\tReference Text\tTranscribed Text
    """
    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if resume and output_tsv.exists():
        with output_tsv.open("r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts:
                    done.add(parts[0])

    write_header = not output_tsv.exists() or output_tsv.stat().st_size == 0

    mode = "a" if output_tsv.exists() else "w"
    with output_tsv.open(mode, encoding="utf-8", newline="") as f_out:
        if write_header:
            f_out.write("Filename\tReference Text\tTranscribed Text\n")

        for row in tqdm(rows, desc="Transcribing"):
            filename = row.filename_stem
            if filename in done:
                continue

            audio_path = clips_dir / row.path
            if not audio_path.exists():
                # Missing file: keep empty transcription (could also be logged).
                transcription = ""
            else:
                try:
                    transcription = transcriber.transcribe_file(audio_path)
                except Exception as e:
                    transcription = f"[ERROR] {type(e).__name__}: {e}"

            ref_text = row.sentence.replace("\t", " ").strip()
            hyp_text = transcription.replace("\t", " ").strip()
            f_out.write(f"{filename}\t{ref_text}\t{hyp_text}\n")
            done.add(filename)
