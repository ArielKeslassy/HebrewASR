from dataclasses import dataclass, field
from collections import Counter
from typing import Iterable, List, Tuple, Dict

@dataclass
class AccuracyStatistics:
    matches_count: int = 0   # #M
    subs_count: int = 0      # #S
    ins_count: int = 0       # #I
    del_count: int = 0       # #D
    error_counter: Counter = field(default_factory=Counter)

    @classmethod
    def from_alignment(cls, alignment: List[Tuple[str, str]]):
        obj = cls()
        obj.add_alignment(alignment)
        return obj

    def add_alignment(self, alignment: List[Tuple[str, str]]):
        for ref_w, hyp_w in alignment:
            if ref_w and hyp_w:
                if ref_w == hyp_w:
                    self.matches_count += 1
                else:
                    self.subs_count += 1
                    self.error_counter[(ref_w, hyp_w)] += 1
            elif ref_w and not hyp_w:
                self.del_count += 1
                self.error_counter[(ref_w, "")] += 1
            elif (not ref_w) and hyp_w:
                self.ins_count += 1
                self.error_counter[("", hyp_w)] += 1

    def __iadd__(self, other: "AccuracyStatistics"):
        self.matches_count += other.matches_count
        self.subs_count += other.subs_count
        self.ins_count += other.ins_count
        self.del_count += other.del_count
        self.error_counter.update(other.error_counter)
        return self 

    @property
    def N_gt(self) -> int:
        return self.matches_count + self.subs_count + self.del_count

    @property
    def N_asr(self) -> int:
        return self.matches_count + self.subs_count + self.ins_count

    @property
    def wer(self) -> float:
        return (self.subs_count + self.ins_count + self.del_count) / self.N_gt if self.N_gt else 0.0

    @property
    def recall(self) -> float:
        return self.matches_count / self.N_gt if self.N_gt else 0.0

    @property
    def precision(self) -> float:
        return self.matches_count / self.N_asr if self.N_asr else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    def frequent_errors(self, min_count: int = 1, top_k: int | None = None):
        items = [(pair, c) for pair, c in self.error_counter.items() if c >= min_count]
        items.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            items = items[:top_k]
        return items

    def to_row(self, filename: str) -> Dict[str, object]:
        return {
            "Filename": filename,
            "N_gt": self.N_gt,
            "N_asr": self.N_asr,
            "#M": self.matches_count,
            "#S": self.subs_count,
            "#I": self.ins_count,
            "#D": self.del_count,
            "WER": self.wer,
            "Recall": self.recall,
            "Precision": self.precision,
            "F1-Score": self.f1,
        }