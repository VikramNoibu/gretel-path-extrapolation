import logging

import numpy as np
import torch
from torch_scatter import scatter_add, scatter_max


def start_idx_from_lengths(lengths):
    """ Compute the starting index of sequences given their length

    Args:
        lengths (torch.LongTensor or numpy.array): lengths of the sub sequences

    Example:
        >>> import torch
        >>> start_idx_from_lengths(torch.tensor([4, 3, 9]))
        tensor([0, 4, 7])

    Returns:
        torch.LongTensor or numpy.array: starting index of the sub sequences
    """
    if lengths.__class__ is torch.Tensor:
        start_idx = torch.zeros_like(lengths)
        start_idx[1:] = torch.cumsum(lengths, dim=0)[:-1]
    else:
        start_idx = np.cumsum(lengths)
        start_idx[1:] = start_idx[:-1]
        start_idx[0] = 0
    return start_idx


def numpify(maybe_tensor):
    """Transform a torch.Tensor into numpy array

    Args:
        maybe_tensor: anything

    Example:
        >>> import torch
        >>> numpify(torch.tensor([1, 2 ,3]))
        array([1, 2, 3])

    Returns:
        numpy.array or anything: numpy array if possible
    """
    if type(maybe_tensor) is torch.Tensor:
        maybe_tensor = maybe_tensor.detach().to("cpu").numpy()
    return maybe_tensor


def sample(elements: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Sample an element from `elements` each with probabity `probabilities`

    Args:
        elements (torch.Tensor): elements to pick from
        probabilities (torch.Tensor): a probability distribution (>= 0 and sum to 1)

    Returns:
        torch.Tensor: one element fo `elements`
    """
    cum_probabilities = torch.cumsum(probabilities, 0)
    prob = torch.rand(1).to(cum_probabilities.device)
    return elements[(cum_probabilities > prob).nonzero()[0, 0]]


def generate_masks(
    trajectory_length,
    number_observations,
    predict="next",
    with_interpolation=False,
    device=None,
):
    """Generate indices mask for observed, start and target indices

    Args:
        trajectory_length: length of the trajectory
        number_observations: number of observed points in the past
        predict (str, optional): Defaults to 'next'. 'next', 'target', 'start_to_target'
        with_interpolation (bool, optional): Defaults to False. use interpolation
        device (torch.Device, optional): Defaults to None. device for the masks

    Examples:
        >>> generate_masks(5, 3)
        (tensor([[0, 1, 2],
                [1, 2, 3]]), tensor([2, 3]), tensor([3, 4]))
        >>> generate_masks(5, 3, with_interpolation=True)
        (tensor([[0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
                [1, 3, 4],
                [1, 2, 4],
                [1, 2, 3]]), tensor([0, 1, 2, 1, 2, 3]), tensor([1, 2, 3, 2, 3, 4]))
        >>> generate_masks(5, 3, predict='start_to_target')
        (tensor([[0, 1, 2]]), tensor([2]), tensor([4]))

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): observed, start, target indices
    """
    assert predict in [
        "next",
        "target",
        "start_to_target",
    ], "`predict` should be one of 'next', 'target', 'start_to_target'"

    n_pred = trajectory_length - number_observations
    if predict == "start_to_target":
        assert not with_interpolation
        observed = torch.arange(number_observations, device=device).unsqueeze(0)
        starts = observed[:, -1]
    elif not with_interpolation:
        # all windows of size num_observations and start at last observation
        observed = torch.arange(number_observations, device=device).unsqueeze(0)
        observed = observed + torch.arange(n_pred, device=device).unsqueeze(1)
        starts = torch.arange(n_pred) + number_observations - 1
    else:
        # all windows of size num_observations and start at each position with next one hidden
        window = torch.arange(number_observations, device=device)
        hide_target = torch.ones(
            [number_observations, number_observations], dtype=torch.long, device=device
        ).triu(1)
        start_delta = torch.arange(n_pred, device=device)

        observed = (
            window.unsqueeze(0) + hide_target + start_delta.view(-1, 1, 1)
        ).view(-1, number_observations)
        starts = (window.unsqueeze(0) + start_delta.unsqueeze(1)).view(-1)

    if predict == "next":
        targets = starts + 1
    elif predict == "target" or predict == "start_to_target":
        assert (
            not with_interpolation
        ), "Should not predict final target for interpolation"
        targets = torch.zeros_like(starts) + trajectory_length - 1

    return observed, starts, targets

def remove_2loops(path):
    """remove loops of lenght 2 from a trajectory
    to be applied after self-loops have been removed
    """
    pattern=set({}) # detect 2 patterns first
    for i in range(len(path[:-3])):
        if path[i]==path[i+2] and path[i+1]==path[i+3]:
            pattern.add((path[i], path[i+1]))
    
    del_index=[] #indices to be removed
    for pat in pattern: #remove each pattern from the list
        count =0
        for i in range(len(path[:-1])):
            if pat[0]==path[i] and pat[1]==path[i+1]: # if pattern match and not the first time, remove them
                if count>0:
                    del_index.extend([i, i+1])
                count = count + 1
    return [path[i] for i in range(len(path)) if i not in del_index]

def remove_self_loops(path):
    """remove repeating consecutive nodes in a path or self-loops"""
    del_index=[] #indices to be removed
    for i in range(len(path[:-1])):
        if path[i]==path[i+1]:
            del_index.append(i)
    return [path[i] for i in range(len(path)) if i not in del_index]