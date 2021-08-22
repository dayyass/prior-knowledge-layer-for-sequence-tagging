from typing import Dict, List, Set

import torch
import torch.nn as nn


def init_prior_knowledge_matrix(
    labels: List[str],
    prohibited_transitions: Dict[str, Set[str]],
    prohibited_transition_value: int = 1,
) -> torch.Tensor:
    """
    Initialization of prior knowledge matrix from labels and prohibited transitions.

    Args:
        labels (List[str]): Labels.
        prohibited_transitions (Dict[str, Set[str]]): Prohibited transitions.
        prohibited_transition_value (int, optional): Value of prohibited transitions. Defaults to 1.

    Returns:
        torch.Tensor: Prior knowledge matrix.
    """

    n = len(labels)
    prior_knowledge_matrix = torch.zeros((n, n), dtype=torch.float)

    for i, label_from in enumerate(labels):
        for j, label_to in enumerate(labels):
            if label_from == label_to:
                continue
            if label_from in prohibited_transitions:
                if label_to in prohibited_transitions[label_from]:
                    prior_knowledge_matrix[i, j] = prohibited_transition_value

    return prior_knowledge_matrix


class PriorKnowledgeLayer(nn.Module):
    """
    Prior Knowledge Layer.
    """

    def __init__(self, prior_knowledge_matrix: torch.Tensor) -> None:
        """
        Prior Knowledge Layer initialization.

        Args:
            prior_knowledge_matrix (torch.Tensor): Prior knowledge matrix.
        """

        super(PriorKnowledgeLayer, self).__init__()

        self.prior_knowledge_matrix = nn.Parameter(prior_knowledge_matrix)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute matrix loss from logits.

        Args:
            logits (torch.Tensor): Logits.

        Returns:
            torch.Tensor: Matrix loss of shape (batch_size, seq_len - 1).
        """

        loss_matrix = _adjacent_reduction_over_seq_len(
            logits=logits,
            prior_knowledge_matrix=self.prior_knowledge_matrix,
        )

        return loss_matrix


def _adjacent_reduction_over_seq_len(
    logits: torch.Tensor,
    prior_knowledge_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Tensor adjacent reduction over seq_len dimension.

    Args:
        logits (torch.Tensor): Logits.
        prior_knowledge_matrix (torch.Tensor): Prior knowledge matrix.

    Returns:
        torch.Tensor: Reduced tensor of shape (batch_size, seq_len - 1).
    """

    distributions = torch.log_softmax(logits, dim=-1)

    batch_size, seq_len, _ = distributions.shape
    adjacent_matrix = torch.zeros((batch_size, seq_len - 1), dtype=torch.float)

    # iterate ower seq_len
    for i in range(seq_len - 1):

        adjacent_reduction = _reduction_over_two_distributions(
            distr_1=distributions[:, i, :],
            distr_2=distributions[:, i + 1, :],
            prior_knowledge_matrix=prior_knowledge_matrix,
        )

        adjacent_matrix[:, i] = adjacent_reduction

    return adjacent_matrix


def _reduction_over_two_distributions(
    distr_1: torch.Tensor,
    distr_2: torch.Tensor,
    prior_knowledge_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Reduction over two distributions and prior knowledge matrix.

    Args:
        distr_1 (torch.Tensor): First distribution.
        distr_2 (torch.Tensor): Second distribution.
        prior_knowledge_matrix (torch.Tensor): Prior knowledge matrix.

    Returns:
        torch.Tensor: Reduction over two distributions.
    """

    return (distr_1 @ prior_knowledge_matrix @ distr_2.T).diag()
