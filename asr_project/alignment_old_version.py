from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class WordEditWeights:
    match_cost: float = 0.0
    substitute_cost: float = 1.0
    insert_cost: float = 1.0
    delete_cost: float = 1.0

    def sub_cost(self, a: str, b: str) -> float:
        return self.match_cost if a == b else self.substitute_cost

def sequences_align(
    ref_words: List[str],
    hyp_words: List[str],
    weights: Optional[WordEditWeights] = None
) -> List[Tuple[str, str]]:
    """
    Returns alignment as list of (ref_word, hyp_word),
    where insertion/deletion represented by "".
    """
    if weights is None:
        weights = WordEditWeights()

    n, m = len(ref_words), len(hyp_words)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]  # M/S/I/D

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + weights.delete_cost
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + weights.insert_cost
        bt[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_c = dp[i - 1][j - 1] + weights.sub_cost(ref_words[i - 1], hyp_words[j - 1])
            del_c = dp[i - 1][j] + weights.delete_cost
            ins_c = dp[i][j - 1] + weights.insert_cost

            best = min(sub_c, del_c, ins_c)
            dp[i][j] = best

            if best == sub_c:
                bt[i][j] = "M" if ref_words[i - 1] == hyp_words[j - 1] else "S"
            elif best == del_c:
                bt[i][j] = "D"
            else:
                bt[i][j] = "I"

    # backtrack
    alignment: List[Tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in ("M", "S"):
            alignment.append((ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            alignment.append((ref_words[i - 1], ""))
            i -= 1
        elif op == "I":
            alignment.append(("", hyp_words[j - 1]))
            j -= 1
        else:
            break

    alignment.reverse()
    return alignment