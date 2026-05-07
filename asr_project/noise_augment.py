from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import random
import math

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import decimate, resample_poly
from tqdm import tqdm

@dataclass
class NoiseProfile:
    source_type: str   # "noise" or "music"
    strength_label: str
    snr_min_db: float
    snr_max_db: float

def profile_from_modulo(modulo_val: int) -> NoiseProfile:
    table = {
        0: NoiseProfile("noise", "strong", 0, 6),
        1: NoiseProfile("music", "strong", 3, 9),
        2: NoiseProfile("noise", "medium", 6, 12),
        3: NoiseProfile("music", "medium", 9, 15),
        4: NoiseProfile("noise", "weak", 12, 18),
        5: NoiseProfile("music", "weak", 15, 21),
    }
    if modulo_val not in table:
        raise ValueError("modulo must be in [0..5]")
    return table[modulo_val]

def _mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32)
    return x.mean(axis=1).astype(np.float32)

def read_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=False)
    x = _mono(np.asarray(x))
    return x, sr

def write_wav(path: Path, x: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr)

def ensure_16k(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if sr == 16000:
        return x, sr
    if sr == 32000:
        # Assignment recommendation: decimate by factor 2.
        y = decimate(x, 2, ftype="iir", zero_phase=True).astype(np.float32)
        return y, 16000
    # Generic fallback for other sample rates.
    y = resample_poly(x, 16000, sr).astype(np.float32)
    return y, 16000

def rms_power(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean(np.square(x)) + eps)

def scale_noise_for_snr(signal: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    p_sig = rms_power(signal)
    p_noise = rms_power(noise)
    target_linear = 10 ** (target_snr_db / 10.0)
    alpha = math.sqrt(p_sig / (p_noise * target_linear))
    return noise * alpha

def _scan_musan_files(profile: NoiseProfile, musan_root: Path) -> List[Path]:
    musan_root = Path(musan_root)

    candidates = []
    if profile.source_type == "music":
        # Recursive search under music folders.
        for p in musan_root.rglob("*.wav"):
            if "music" in str(p).lower():
                candidates.append(p)
    else:
        # Noise only; assignment asks to use long noises (>30s), typically under free-sound/noise.
        # Folder naming varies across MUSAN versions, so search "noise" and filter by duration.
        for p in musan_root.rglob("*.wav"):
            if "noise" in str(p).lower():
                candidates.append(p)

    return sorted(set(candidates))

def _filter_long_noise(files: List[Path], min_seconds: float = 30.0) -> List[Path]:
    good = []
    for p in files:
        try:
            info = sf.info(str(p))
            dur = info.frames / info.samplerate
            if dur >= min_seconds:
                good.append(p)
        except Exception:
            continue
    return good

def create_noisy_benchmark(
    source_audio_dir: Path,
    filenames: List[str],            # stems only
    original_ext_lookup: dict,       # stem -> filename.ext (e.g., "abc" -> "abc.mp3")
    musan_root: Path,
    output_noisy_dir: Path,
    output_log_tsv: Path,
    profile: NoiseProfile,
    seed: int = 42,
):
    rng = random.Random(seed)
    output_noisy_dir = Path(output_noisy_dir)
    output_noisy_dir.mkdir(parents=True, exist_ok=True)

    bg_files = _scan_musan_files(profile, musan_root)
    if profile.source_type == "noise":
        bg_files = _filter_long_noise(bg_files, min_seconds=30.0)

    if not bg_files:
        raise RuntimeError("No MUSAN background files found for selected profile")

    log_rows = []

    for stem in tqdm(filenames, desc="Creating noisy benchmark"):
        src_name = original_ext_lookup[stem]
        src_path = Path(source_audio_dir) / src_name

        try:
            speech, sr_s = read_audio_mono(src_path)
            speech, sr_s = ensure_16k(speech, sr_s)

            bg_path = rng.choice(bg_files)
            bg, sr_b = read_audio_mono(bg_path)
            bg, sr_b = ensure_16k(bg, sr_b)

            if sr_s != 16000 or sr_b != 16000:
                raise RuntimeError("Sampling rate mismatch after normalization")

            if len(bg) < len(speech):
                # Repeat background if it is shorter than speech.
                reps = int(np.ceil(len(speech) / len(bg)))
                bg = np.tile(bg, reps)

            max_start = len(bg) - len(speech)
            start_idx = rng.randint(0, max_start) if max_start > 0 else 0
            bg_seg = bg[start_idx:start_idx + len(speech)]

            snr_db = rng.uniform(profile.snr_min_db, profile.snr_max_db)
            bg_scaled = scale_noise_for_snr(speech, bg_seg, snr_db)
            noisy = speech + bg_scaled

            # clipping protection
            peak = np.max(np.abs(noisy)) + 1e-12
            if peak > 1.0:
                noisy = noisy / peak * 0.98

            out_path = output_noisy_dir / f"{stem}.wav"
            write_wav(out_path, noisy.astype(np.float32), 16000)

            log_rows.append({
                "Filename": stem,
                "Background file": str(bg_path),
                "Start point (in seconds)": start_idx / 16000.0,
                "SNR": snr_db,
            })

        except Exception as e:
            log_rows.append({
                "Filename": stem,
                "Background file": f"[ERROR] {type(e).__name__}: {e}",
                "Start point (in seconds)": "",
                "SNR": "",
            })

    pd.DataFrame(log_rows).to_csv(output_log_tsv, sep="\t", index=False, encoding="utf-8")
