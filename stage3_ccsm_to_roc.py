import ast
import random
import re
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pydot

from stage2_roc_to_csm import Event, State, StateMachine

def get_neighbors_old(matrix, row, col):
    neighbors = []
    if col > 0:
        neighbors.append((row, col - 1))
    if col < matrix.shape[1] - 1:
        neighbors.append((row, col + 1))
    if row > 0:
        neighbors.append((row - 1, col))
    if row < matrix.shape[0] - 1:
        neighbors.append((row + 1, col))
    
    return neighbors
    
def find_min_neighbor(matrix, loc):
    row,col = loc
    neighbors = get_neighbors_old(matrix, row, col)
    min_val = min(matrix[i, j] for i, j in neighbors)
    min_neighbors = [(i, j) for i, j in neighbors if matrix[i, j] == min_val]
    return random.choice(min_neighbors)

def get_action(loc, dest):
    # print(loc)
    # print("\n",dest)
    row, col = loc
    dest_row, dest_col = dest
    
    if dest_row == row and dest_col == col + 1:
        return 0  # Right
    elif dest_row == row - 1 and dest_col == col:
        return 1  # Top
    elif dest_row == row and dest_col == col - 1:
        return 2  # Left
    elif dest_row == row + 1 and dest_col == col:
        return 3  # Bottom
    else:
        return 4  # Stay
    
def get_obligation_from_cross_csm(sm):
    """
    Returns the abstract role obligation stored in the cross-scenario CSM.
    Expected format:
        sm.role_obligation = {
            "type": "Obl",
            "target": "Step Clockwise" or "Contour(M)",
            "duration": delta_R
        }
    """
    return getattr(sm, "role_obligation", None)

def clockwise_initial_step(loc, cellvalue, goal):
    """
    Minimal version:
    choose among lowest-value neighbours, then take the first one
    in a fixed clockwise priority order.

    Action order:
        right, down, left, up
    """
    row, col = loc

    clockwise_candidates = [
        (row, col + 1),  # right
        (row + 1, col),  # down
        (row, col - 1),  # left
        (row - 1, col),  # up
    ]

    valid_candidates = [
        c for c in clockwise_candidates
        if 0 <= c[0] < cellvalue.shape[0]
        and 0 <= c[1] < cellvalue.shape[1]
        and cellvalue[c] != -100
    ]

    if not valid_candidates:
        dest = find_min_neighbor(cellvalue, loc)
        return get_action(loc, dest), dest

    min_value = min(cellvalue[c] for c in valid_candidates)

    best_candidates = [
        c for c in valid_candidates
        if cellvalue[c] == min_value
    ]

    # Because valid_candidates is already ordered clockwise,
    # this chooses the first clockwise optimal option.
    dest = best_candidates[0]

    return get_action(loc, dest), dest

def contour_step(loc, cellvalue, contour_cells):
    """
    Choose the lowest-value neighbour that remains inside the required contour.
    If no contour neighbour is available, fall back to ordinary optimal movement.
    """
    row, col = loc
    neighbors = get_neighbors_old(cellvalue, row, col)

    allowed_neighbors = [
        n for n in neighbors
        if n in contour_cells and cellvalue[n] != -100
    ]

    if not allowed_neighbors:
        dest = find_min_neighbor(cellvalue, loc)
        return get_action(loc, dest), dest

    min_value = min(cellvalue[n] for n in allowed_neighbors)

    best_neighbors = [
        n for n in allowed_neighbors
        if cellvalue[n] == min_value
    ]

    dest = random.choice(best_neighbors)

    return get_action(loc, dest), dest