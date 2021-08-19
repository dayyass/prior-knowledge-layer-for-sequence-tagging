from typing import Dict, List, Set

import numpy as np


def init_prior_knowledge_matrix(
    labels: List[str],
    prohibited_transitions: Dict[str, Set[str]],
    prohibited_transition_value: int = 1,
) -> np.ndarray:
    """Initialization of prior knowledge matrix from labels and prohibited transitions.

    Args:
        labels (List[str]): Labels.
        prohibited_transitions (Dict[str, Set[str]]): Prohibited transitions.
        prohibited_transition_value (int, optional): Value of prohibited transitions. Defaults to 1.

    Returns:
        np.ndarray: Prior knowledge matrix.
    """

    n = len(labels)
    prior_knowledge_matrix = np.zeros(shape=(n, n), dtype="float32")

    for i, label_from in enumerate(labels):
        for j, label_to in enumerate(labels):
            if label_from == label_to:
                continue
            if label_from in prohibited_transitions:
                if label_to in prohibited_transitions[label_from]:
                    prior_knowledge_matrix[i, j] = prohibited_transition_value

    return prior_knowledge_matrix
