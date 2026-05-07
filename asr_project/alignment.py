from dataclasses import dataclass
from typing import List, Tuple, Optional

from sequence_alignment import align_sequences, EditWeights


@dataclass
class WordEditWeights:
    match_cost: float = 0.0
    substitute_cost: float = 1.0
    insert_cost: float = 1.0
    delete_cost: float = 1.0


class _MaximizationWeightsAdapter(EditWeights):
    """
    The provided aligner maximizes a score, while WER-style alignment minimizes edit costs.
    We convert costs to negative weights, so maximizing the weight is equivalent to minimizing cost.
    """

    def __init__(self, costs: WordEditWeights):
        self.costs = costs

    def pair_weight(self, first_obj, second_obj) -> float:
        if first_obj == second_obj:
            return -float(self.costs.match_cost)
        return -float(self.costs.substitute_cost)

    def insertion_weight(self, obj) -> float:
        return -float(self.costs.insert_cost)

    def deletion_weight(self, obj) -> float:
        return -float(self.costs.delete_cost)


def align_word_sequences(
    ref_words: List[str],
    hyp_words: List[str],
    weights: Optional[WordEditWeights] = None,
) -> Tuple[List[Tuple[str, str]], float]:
    if weights is None:
        weights = WordEditWeights()

    ref_words = list(ref_words)
    hyp_words = list(hyp_words)

    score, aligned_pairs = align_sequences(
        ref_words,
        hyp_words,
        _MaximizationWeightsAdapter(weights),
    )

    alignment = [((a if a is not None else ""), (b if b is not None else "")) for a, b in aligned_pairs]
    return alignment, float(score)