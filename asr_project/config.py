from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProjectConfig:
    project_root: Path
    commonvoice_root: Path
    musan_root: Path
    outputs_dir: Path
    noisy_dir: Path

    # IvritAI Whisper CT2 model
    model_name_or_path: str = "ivrit-ai/whisper-large-v3-turbo-ct2"
    language: str = "he"
    device: str = "cpu"         # "cuda" if GPU available
    compute_type: str = "int8"  # CPU: int8 ; GPU often float16

    @property
    def commonvoice_test_tsv(self) -> Path:
        return self.commonvoice_root / "test.tsv"

    @property
    def commonvoice_clips_dir(self) -> Path:
        return self.commonvoice_root / "clips"

    @classmethod
    def default(cls, project_root: Path):
        project_root = Path(project_root)
        return cls(
            project_root=project_root,
            commonvoice_root=project_root / "data" / "commonvoice_he",
            musan_root=project_root / "data" / "musan",
            outputs_dir=project_root / "data" / "outputs",
            noisy_dir=project_root / "data" / "noisy_benchmark",
        )

    def ensure_dirs(self):
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.noisy_dir.mkdir(parents=True, exist_ok=True)