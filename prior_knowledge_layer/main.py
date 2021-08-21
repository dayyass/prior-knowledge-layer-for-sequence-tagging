from typing import Dict, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            torch.Tensor: Matrix loss of shape (batch_size, seq_len-1).
        """

        log_softmax = F.log_softmax(logits, dim=-1)

        timestamp_loss_list = []

        # iterate ower seq_len
        for i in range(log_softmax.shape[1] - 1):
            timestamp_loss = (
                log_softmax[:, i, :]
                @ self.prior_knowledge_matrix
                @ log_softmax[:, i + 1, :].T
            ).diag()
            timestamp_loss_list.append(timestamp_loss)

        # shape (batch_size, seq_len-1)
        loss_matrix = torch.stack(timestamp_loss_list, dim=-1)
        return loss_matrix
