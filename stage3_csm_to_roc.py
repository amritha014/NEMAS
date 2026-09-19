import ast
import random
import re
import sys

import shapely
from shapely import MultiPoint

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pydot

from stage2_roc_to_csm import Event, State, StateMachine

DEST_A = None
DEST_B = None
ACTION_A = None
ACTION_B = None

ACTION_DELTAS = {
    0: (0, 1),    # right
    1: (-1, 0),   # up
    2: (0, -1),   # left
    3: (1, 0),    # down
    4: (0, 0)     # stay
}

def get_orthogonal_contour_neighbours(cell, contour_cells):
    """
    Return the cells in contour_cells that are directly above, below,
    left or right of the given cell.
    """
    row, col = cell

    possible_neighbours = [
        (row - 1, col),  # up
        (row + 1, col),  # down
        (row, col - 1),  # left
        (row, col + 1)   # right
    ]

    return [
        neighbour
        for neighbour in possible_neighbours
        if neighbour in contour_cells
    ]


def validate_contour_cycle(contour_cells):
    """
    Check that the contour is a single grid cycle.

    In a simple closed grid cycle, every contour cell must have exactly
    two orthogonally adjacent neighbours that are also in the contour.
    """
    contour_cells = set(contour_cells)

    if len(contour_cells) < 4:
        raise ValueError(
            "A contour cycle must contain at least four cells."
        )

    for cell in contour_cells:
        neighbours = get_orthogonal_contour_neighbours(
            cell,
            contour_cells
        )

        if len(neighbours) != 2:
            raise ValueError(
                f"Contour is not a simple closed cycle. "
                f"Cell {cell} has {len(neighbours)} contour neighbours: "
                f"{neighbours}. A cycle cell must have exactly two."
            )


def get_clockwise_contour_cycle(contour_cells):
    """
    Convert an unordered set of contour cells into a clockwise sequence.

    Shapely first creates a concave hull around the points. The exterior
    ring is then normalised so that its coordinates follow Shapely's
    canonical exterior-ring orientation.
    """
    contour_cells = set(contour_cells)
    print("\n========== CLOCKWISE CONTOUR GENERATION ==========")
    print("Original unordered contour cells:")
    print(sorted(contour_cells))

    validate_contour_cycle(contour_cells)

    multipoint = MultiPoint(list(contour_cells))

    print("\nShapely MultiPoint:")
    print(multipoint)

    hull = shapely.concave_hull(multipoint)

    print("\nConcave hull:")
    print(hull)

    if hull.is_empty:
        raise ValueError(
            "Shapely could not construct a hull from the contour cells."
        )

    if hull.geom_type != "Polygon":
        raise ValueError(
            f"Expected the contour hull to be a Polygon, "
            f"but Shapely produced {hull.geom_type}."
        )

    exterior_ring = hull.exterior

    print("\nExterior ring before normalisation:")
    print(exterior_ring)

    print("\nExterior-ring coordinates before normalisation:")
    print(list(exterior_ring.coords))

    normalised_ring = shapely.normalize(exterior_ring)

    print("\nNormalised exterior ring:")
    print(normalised_ring)

    print("\nNormalised ring coordinates:")
    print(list(normalised_ring.coords))

    ordered_cells = [
        (int(row), int(col))
        for row, col in normalised_ring.coords
    ]

    # A LinearRing repeats its first coordinate at the end.
    # Remove the repeated coordinate before treating the list as a cycle.
    if (
        len(ordered_cells) > 1
        and ordered_cells[0] == ordered_cells[-1]
    ):
        ordered_cells = ordered_cells[:-1]

    print("\nClockwise cell cycle after removing repeated cell:")
    for index, cell in enumerate(ordered_cells):
        print(f"Index {index}: {cell}")

    # Check that Shapely has retained all contour cells.
    missing_cells = contour_cells.difference(ordered_cells)
    extra_cells = set(ordered_cells).difference(contour_cells)

    if missing_cells or extra_cells:
        raise ValueError(
            "The Shapely exterior ring does not exactly match the "
            "provided contour cells. "
            f"Missing cells: {missing_cells}; "
            f"unexpected cells: {extra_cells}."
        )

    # Check that consecutive ring cells are valid grid neighbours.
    for index, current_cell in enumerate(ordered_cells):
        next_cell = ordered_cells[
            (index + 1) % len(ordered_cells)
        ]

        distance = (
            abs(next_cell[0] - current_cell[0])
            + abs(next_cell[1] - current_cell[1])
        )

        if distance != 1:
            raise ValueError(
                "The normalised Shapely ring contains consecutive "
                "coordinates that are not orthogonally adjacent: "
                f"{current_cell} -> {next_cell}."
            )

    return ordered_cells


def movement_delta_to_action(current_cell, next_cell):
    """
    Convert movement between two adjacent cells into an action number.
    """
    movement_delta = (
        next_cell[0] - current_cell[0],
        next_cell[1] - current_cell[1]
    )

    for action, action_delta in ACTION_DELTAS.items():
        if action_delta == movement_delta:
            return action

    raise ValueError(
        f"Movement from {current_cell} to {next_cell} "
        f"does not correspond to a valid grid action. "
        f"Movement delta: {movement_delta}."
    )

def choose_clockwise_contour_action(
    global_loc,
    value_matrix,
    contour_cells,
    contour_label,
    rowstartidx,
    colstartidx
):
    """
    Select the action that moves the agent to the next cell in the
    clockwise ordering of the required contour.

    global_loc is converted into local coordinates because contour_cells
    are represented using local matrix coordinates.
    """
    local_loc = global_to_local(
        global_loc,
        rowstartidx,
        colstartidx
    )

    required_contour_cells = (
        set(contour_cells.get(contour_label, set()))
        if contour_cells
        else set()
    )

    if not required_contour_cells:
        raise ValueError(
            f"No cells were supplied for Contour({contour_label})."
        )

    if local_loc not in required_contour_cells:
        raise ValueError(
            f"The agent's current local cell {local_loc} is not on "
            f"Contour({contour_label}). "
            f"Contour cells are: {required_contour_cells}"
        )

    clockwise_cells = get_clockwise_contour_cycle(
        required_contour_cells
    )

    print("\n========== CLOCKWISE MOVE SELECTION ==========")
    print(f"Contour label: {contour_label}")
    print(f"Agent global location: {global_loc}")
    print(f"Agent local location: {local_loc}")
    print(f"Clockwise ring: {clockwise_cells}")

    current_index = clockwise_cells.index(local_loc)

    print(f"Current cell index in ring: {current_index}")
    next_index = (
        current_index + 1
    ) % len(clockwise_cells)

    next_local_cell = clockwise_cells[next_index]

    print(f"Next clockwise cell: {next_local_cell}")

    action = movement_delta_to_action(
        local_loc,
        next_local_cell
    )

    print(f"Selected action number: {action}")
    print(f"Action delta: {ACTION_DELTAS[action]}")
    print("==============================================\n")

    next_global_cell = local_to_global(
        next_local_cell,
        rowstartidx,
        colstartidx
    )

    print(
        f"Clockwise Contour({contour_label}) move: "
        f"{local_loc} -> {next_local_cell}; "
        f"action = {action}"
    )

    return action, next_global_cell

def local_to_global(local_coord, rowstartidx, colstartidx):
    r, c = local_coord
    return (r + rowstartidx, c + colstartidx)


def global_to_local(global_coord, rowstartidx, colstartidx):
    r, c = global_coord
    return (r - rowstartidx, c - colstartidx)


def coord_to_state(global_coord, max_cols=16):
    r, c = global_coord
    return r * max_cols + c


def apply_action(coord, action):
    dr, dc = ACTION_DELTAS[action]
    return (coord[0] + dr, coord[1] + dc)


def is_inside_matrix(coord, matrix):
    r, c = coord
    return 0 <= r < matrix.shape[0] and 0 <= c < matrix.shape[1]


def get_policy_action(global_loc, optimal_policy_for_agent):
    state = coord_to_state(global_loc, max_cols=16)
    return optimal_policy_for_agent[state]


def choose_lowest_value_action(candidate_actions, value_matrix):
    best_value = min(
        value_matrix[dest[0], dest[1]]
        for action, dest in candidate_actions
    )

    best_actions = [
        (action, dest)
        for action, dest in candidate_actions
        if value_matrix[dest[0], dest[1]] == best_value
    ]

    return random.choice(best_actions)


def choose_suboptimal_m_action(
    global_loc,
    value_matrix,
    contour_cells,
    optimal_policy_for_agent,
    rowstartidx,
    colstartidx
):
    local_loc = global_to_local(global_loc, rowstartidx, colstartidx)
    optimal_action = get_policy_action(global_loc, optimal_policy_for_agent)

    m_cells = contour_cells.get("M", set()) if contour_cells else set()

    candidate_actions = []

    for action in [0, 1, 2, 3]:
        local_dest = apply_action(local_loc, action)

        if not is_inside_matrix(local_dest, value_matrix):
            continue

        if local_dest in m_cells and action != optimal_action:
            candidate_actions.append((action, local_dest))

    if candidate_actions:
        action, local_dest = choose_lowest_value_action(candidate_actions, value_matrix)
        return action, local_to_global(local_dest, rowstartidx, colstartidx)

    raise ValueError(
        f"No valid suboptimal M-contour move found from {local_loc}. "
        f"M cells are: {m_cells}"
    )


def choose_contour_m_action(
    global_loc,
    value_matrix,
    contour_cells,
    optimal_policy_for_agent,
    rowstartidx,
    colstartidx
):
    local_loc = global_to_local(global_loc, rowstartidx, colstartidx)
    optimal_action = get_policy_action(global_loc, optimal_policy_for_agent)

    m_cells = contour_cells.get("M", set()) if contour_cells else set()

    # Prefer the optimal action if it keeps the agent inside M.
    optimal_local_dest = apply_action(local_loc, optimal_action)

    if (
        is_inside_matrix(optimal_local_dest, value_matrix)
        and optimal_local_dest in m_cells
    ):
        return optimal_action, local_to_global(
            optimal_local_dest,
            rowstartidx,
            colstartidx
        )

    # Otherwise choose another action that moves into/stays in M.
    candidate_actions = []

    for action in [0, 1, 2, 3]:
        local_dest = apply_action(local_loc, action)

        if not is_inside_matrix(local_dest, value_matrix):
            continue

        if local_dest in m_cells:
            candidate_actions.append((action, local_dest))

    if candidate_actions:
        action, local_dest = choose_lowest_value_action(candidate_actions, value_matrix)
        return action, local_to_global(local_dest, rowstartidx, colstartidx)

    raise ValueError(
        f"No valid Contour(M) move found from {local_loc}. "
        f"M cells are: {m_cells}"
    )

def choose_step_to_and_stay_in_m_action(
    global_loc,
    value_matrix,
    contour_cells,
    optimal_policy_for_agent,
    rowstartidx,
    colstartidx
):
    local_loc = global_to_local(
        global_loc,
        rowstartidx,
        colstartidx
    )

    m_cells = contour_cells.get("M", set()) if contour_cells else set()

    # Agent has not entered contour M yet.
    if local_loc not in m_cells:
        return choose_suboptimal_m_action(
            global_loc,
            value_matrix,
            contour_cells,
            optimal_policy_for_agent,
            rowstartidx,
            colstartidx
        )

    # Agent is already in contour M.
    # Keep moving within contour M.
    return choose_contour_m_action(
        global_loc,
        value_matrix,
        contour_cells,
        optimal_policy_for_agent,
        rowstartidx,
        colstartidx
    )

def execute_obligation(
    global_loc,
    value_matrix,
    obligation,
    contour_cells,
    optimal_policy_for_agent,
    rowstartidx,
    colstartidx
):
    target = obligation.get("target")

    if target == "Contour(M)":
        return choose_step_to_and_stay_in_m_action(
            global_loc,
            value_matrix,
            contour_cells,
            optimal_policy_for_agent,
            rowstartidx,
            colstartidx
        )
    elif target.startswith("Step Clockwise Contour("):
        contour_label = target[len("Step Clockwise Contour("):-1]

        return choose_clockwise_contour_action(
            global_loc,
            value_matrix,
            contour_cells,
            contour_label,
            rowstartidx,
            colstartidx
        )

    else:
        action = get_policy_action(global_loc, optimal_policy_for_agent)
        return action, apply_action(global_loc, action)
    


def obligation_completed(completed_steps, obligation):
    """Return True when the current state's obligation duration is satisfied."""
    required_steps = obligation.get("duration", 1)
    return completed_steps >= required_steps


def dispatch_runtime_event(sm, event_name):
    """
    Dispatch through the CCSM itself.

    StateMachineWithConstraints provides dispatch_event(). The fallback keeps
    this executor compatible with a plain pysm StateMachine.
    """
    if hasattr(sm, "dispatch_event"):
        sm.dispatch_event(event_name)
    else:
        sm.dispatch(Event(event_name))


def csm_to_roc(
    sm,
    start_coord_A,
    end_coord_A,
    cellvalue_A,
    start_coord_B,
    end_coord_B,
    cellvalue_B,
    agent,
    contour_cells,
    optimal_policy_for_agent,
    rowstartidx,
    colstartidx,
    max_steps=200
):
    if agent == "A":
        local_loc = start_coord_A
        goal_global = end_coord_A
        value_matrix = cellvalue_A
    else:
        local_loc = start_coord_B
        goal_global = end_coord_B
        value_matrix = cellvalue_B

    global_loc = local_to_global(local_loc, rowstartidx, colstartidx)

    plan = []
    steps = 0

    # Use the state machine's actual current-state mechanism.
    sm.reset()
    if hasattr(sm, "initialize"):
        sm.initialize()

    print("CCSM State Machine Initialized")
    print("Initial state:", sm.state.name)
    print("Start local:", local_loc)
    print("Start global:", global_loc)
    print("Goal global:", goal_global)

    while sm.state.name != "end":
        if steps >= max_steps:
            raise RuntimeError("CCSM execution exceeded max_steps.")

        current_state = sm.state

        print("---------- CCSM STATE ----------")
        print("State:", current_state.name)
        print("Global location:", global_loc)
        print(
            "Local location:",
            global_to_local(global_loc, rowstartidx, colstartidx)
        )
        print("Constraints:", current_state.constraints)

        # The goal event is recognised from the runtime location.
        if global_loc == goal_global:
            dispatch_runtime_event(sm, "goal_reached")
            continue

        cons = current_state.constraints

        if cons and "Obl" in cons:
            obligation = cons["Obl"]
            completed_steps = 0

            # Stay in the same CCSM state until its obligation is completed.
            while not obligation_completed(completed_steps, obligation):
                if steps >= max_steps:
                    raise RuntimeError("CCSM execution exceeded max_steps.")

                if global_loc == goal_global:
                    break

                action, next_global = execute_obligation(
                    global_loc,
                    value_matrix,
                    obligation,
                    contour_cells,
                    optimal_policy_for_agent,
                    rowstartidx,
                    colstartidx
                )

                print("Obligation:", obligation)
                print("Action:", action)
                print("Move:", global_loc, "->", next_global)

                plan.append(action)
                global_loc = next_global
                completed_steps += 1
                steps += 1

            if global_loc == goal_global:
                dispatch_runtime_event(sm, "goal_reached")
            elif obligation_completed(completed_steps, obligation):
                # Required CCSM release event after regret_con/noregret_con.
                dispatch_runtime_event(sm, "rel")

        else:
            # Unconstrained CCSM state: take one optimal-policy action.
            action = get_policy_action(global_loc, optimal_policy_for_agent)
            next_global = apply_action(global_loc, action)

            print("No obligation: following optimal policy")
            print("Action:", action)
            print("Move:", global_loc, "->", next_global)

            plan.append(action)
            global_loc = next_global
            steps += 1

            if global_loc == goal_global:
                dispatch_runtime_event(sm, "goal_reached")

    print(f"Agent {agent} reached the CCSM end state.")
    return plan