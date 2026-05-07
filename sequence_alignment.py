from typing import Iterable, List, Tuple
import numpy as np

class EditWeights:
    """
    Abstract base class for classes providing edit weights for sequence alignment operations.
    """

    def pair_weight(self, first_obj, second_obj) -> float:
        """
        Get the weight for aligning the pair of given objects.

        @param first_obj The object from the first sequence.
        @param second_obj The object from the second sequence.
        @return The match weight if the two objects are equal,
                otherwise the substitution weight.
        """
        pass

    def insertion_weight(self, obj) -> float:
        """
        Get the insertion weight of the given object.

        @param obj The inserted object in the second sequence.
        @return The insertion weight.
        """
        pass

    def deletion_weight(self, obj) -> float:
        """
        Get the deletion weight of the given object.

        @param obj The deleted object in the first sequence.
        @return The deletion weight.
        """
        pass

def align_sequences(first_seq: Iterable,
                    second_seq: Iterable,
                    weights: EditWeights) -> Tuple[float, List[Tuple]]:
    """
    Perform global alignment between the two given sequences and compute the score of the optimal alignment with respect
    to the given edit weights.

    If the lengths of the input sequences are m and n, this function takes O(mn) time and uses O(mn) memory.

    @param first_seq The first sequence of aligned objects.
    @param second_seq The second sequence of aligned objects.
    @param weights The edit weights.
    @return The computed score of the optimal global alignment.
    @return An output list of the aligned object pairs:
            The list contains pairs of the form (a_i, b_j) of objects from the first and the second input
            sequence, respectively. The pair (None, b_j) represents object insertion in the second
           sequence, and the pair (a_i, None) represents object deletion from the first sequence.
    """

    # Codes for the various edit operations:
    _OP_NULL = 0
    _OP_PAIR = 1
    _OP_INS = 2
    _OP_DEL = 3

    # Create the matrix of alignment scores and the matrix of back pointers.
    first_len = len(first_seq)
    second_len = len(second_seq)

    scores_mat = np.zeros((first_len + 1, second_len + 1))
    ops_mat = np.zeros((first_len + 1, second_len + 1), dtype=np.int8)

    # Set up the initial matrix entry.
    scores_mat[0, 0] = 0
    ops_mat[0, 0] = _OP_NULL

    # Fill the 0'th row, corresponding to the insertion of the objects in the second sequence.
    j = 1
    for second_obj in second_seq:
        # Sum up the insertion weight of the current object in the second sequence with the predecessor entry.
        ins_wgt = scores_mat[0, j - 1] + weights.insertion_weight(second_obj)

        scores_mat[0, j] = ins_wgt
        ops_mat[0, j] = _OP_INS

        j += 1

    # Fill the rest of the matrix: Go over all objects of the first sequence.
    i = 1
    for first_obj in first_seq:
        # The 0'th column corresponds to the deletion of the objects in the first sequence.
        del_wgt = scores_mat[i - 1, 0] + weights.deletion_weight(first_obj)

        scores_mat[i, 0] = del_wgt
        ops_mat[i, 0] = _OP_DEL

        # Fill the rest of the row: Go over all objects of the second sequence.
        j = 1
        for second_obj in second_seq:
            # Accumulate the match or substitution weight with the weight of the corresponding predecessor entry.
            max_wgt = scores_mat[i - 1, j - 1] + weights.pair_weight(first_obj, second_obj)
            best_op = _OP_PAIR

            # Accumulate the current insertion weight with the weight of the corresponding predecessor entry.
            ins_wgt = scores_mat[i, j - 1] + weights.insertion_weight(second_obj)
            if ins_wgt > max_wgt:
                max_wgt = ins_wgt
                best_op = _OP_INS

            # Accumulate the current deletion weight with the weight of the corresponding predecessor entry.
            del_wgt = scores_mat[i - 1, j] + weights.deletion_weight(first_obj)
            if del_wgt > max_wgt:
                max_wgt = del_wgt
                best_op = _OP_DEL

            # Store the selected maximum weight and its corresponding operation.
            scores_mat[i, j] = max_wgt
            ops_mat[i, j] = best_op

            # Proceed to the next object in the second sequence.
            j += 1

        # Proceed to the next object in the first sequence.
        i += 1

    # Start with an empty list of aligned pairs, and from the given row and column.
    aligned_pairs = []
    i = first_len
    j = second_len

    while i > 0 or j > 0:
        # Check the edit operation at the current entry and act accordingly.
        curr_op = ops_mat[i, j]

        if curr_op == _OP_PAIR:
            # Pair operation: go back to the previous row and the previous column.
            i -= 1
            j -= 1
            aligned_pairs.append((first_seq[i], second_seq[j]))

        elif curr_op == _OP_INS:
            # Insertion operation: go back to the previous column in the same row.
            j -= 1
            aligned_pairs.append((None, second_seq[j]))

        elif curr_op == _OP_DEL:
            # Deletion operation: go back to the previous row in the same column.
            i -= 1
            aligned_pairs.append((first_seq[i], None))

        else:
            break

    # As we traced the alignment backwards, we have to reverse the list of aligned pairs.
    aligned_pairs.reverse()

    # Return the global alignment score and the aligned pairs.
    return scores_mat[first_len, second_len], aligned_pairs

