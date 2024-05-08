import numpy as np
from copy import deepcopy
from typing import List, Union, Optional


def mergesort(
    ixs: List[int],
    P: np.ndarray,
    beam: bool,
    beam_size: Optional[int]=None,
    Uh: Optional[float]=None,
    U: Optional[np.ndarray]=None
) -> List[int]:
    '''
    Perform either PairS-greedy (standard mergesort), or PairS-beam.
    '''

    # if using beam, we need all hyperparameters
    if beam:
        for var in [beam_size, Uh, U]: assert var is not None

    # base case
    if len(ixs) <= 1: return ixs

    # standard mergesort recursion, optionally using beam
    mid = len(ixs) // 2
    left, right = ixs[:mid], ixs[mid:]
    left = mergesort(left, P, beam, beam_size, Uh, U)
    right = mergesort(right, P, beam, beam_size, Uh, U)

    if not beam:
        return merge(left, right, P)
    else:
        return merge_beam(left, right, P, beam_size, Uh, U)

def merge(
    left: List[int],
    right: List[int],
    P: np.ndarray
) -> List[int]:
    '''
    Standard merging.
    '''

    merged = []; i = 0; j = 0
    while i < len(left) and j < len(right):
        # check if P(i >- j)
        if P[left[i], right[j]] > 0.5:
            merged.append(left[i]); i += 1
        else:
            # note: in this setting if P is not symmetric 
            # we will default to this branch of the if-statement under too much uncertainty
            merged.append(right[j]); j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

def merge_beam(
    left: List[int],
    right: List[int],
    P: np.ndarray,
    beam_size: int,
    Uh: float,
    U: np.ndarray
) -> List[int]:
    '''
    `Algorithm 1` in https://arxiv.org/abs/2403.16950

    Instead of maximising L_t we will maximise log-likelihood.
    '''

    L, R = len(left), len(right)
    # init beam with empty candidate
    # stores trajectory, pointers to left and right, log-likelihood
    B = [{"traj": [], "i": 0, "j": 0, "ll": 0}]
    for k in range(1, L+R+1):
        # new beam
        B_ = []

        # extend each original candidate
        for candidate in B:
            i, j = candidate["i"], candidate["j"]
            if i < L and j < R:
                if U[left[i], right[j]] > Uh:
                    # too much uncertainty - add both choices as candidates
                    c = deepcopy(candidate)
                    c["traj"].append(left[i])
                    c["i"] += 1
                    c["ll"] += np.log(P[left[i], right[j]] + 1e-10)
                    B_.append(c)

                    c = deepcopy(candidate)
                    c["traj"].append(right[j])
                    c["j"] += 1
                    c["ll"] += np.log(P[right[j], left[i]] + 1e-10)
                    B_.append(c)

                # below uses two if statements
                # i.e., we check both P(i,j) and P(j,i)
                # this is to avoid assuming that P is symmetric
                #
                # this differs from the original PairS implementation!
                # note this means we need a failure condition if neither of these trigger
                confident = False
                if P[left[i], right[j]] > 0.5:
                    confident = True
                    c = deepcopy(candidate)
                    c["traj"].append(left[i])
                    c["i"] += 1
                    c["ll"] += np.log(P[left[i], right[j]] + 1e-10)
                    B_.append(c)
                if P[right[j], left[i]] > 0.5:
                    confident = True
                    c = deepcopy(candidate)
                    c["traj"].append(right[j])
                    c["j"] += 1
                    c["ll"] += np.log(P[right[j], left[i]] + 1e-10)
                    B_.append(c)
                if not confident:
                    # too much uncertainty - add both choices as candidates
                    c = deepcopy(candidate)
                    c["traj"].append(left[i])
                    c["i"] += 1
                    c["ll"] += np.log(P[left[i], right[j]] + 1e-10)
                    B_.append(c)

                    c = deepcopy(candidate)
                    c["traj"].append(right[j])
                    c["j"] += 1
                    c["ll"] += np.log(P[right[j], left[i]] + 1e-10)
                    B_.append(c)

            elif i < L:
                # we still have elements on the left to add
                candidate["traj"].append(left[i])
                candidate["i"] += 1
                B_.append(candidate)
            elif j < R:
                # we still have elements on the right to add
                candidate["traj"].append(right[j])
                candidate["j"] += 1
                B_.append(candidate)

        # sort the new candidates by likelihood
        B_.sort(key=lambda c: c["ll"], reverse=True)
        # select only the top n candidates
        B = B_[:beam_size]
    # return the trajectory of the top candidate
    return B[0]["traj"]