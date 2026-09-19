import ast
import pickle
import random
import re
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pydot
from pysm import Event, State, StateMachine


class ConstrainedState(State):
    def __init__(self, name,cellcoord=None, Tstep=None, transitions=None):
        super().__init__(name)
        self.constraints = {}
        self.Tstep =Tstep
        self.cellcoord = cellcoord
        self.transitions = []

    def __str__(self):
        state_str = f"{self.name}:\n"
        state_str += f"  Constraints: {self.constraints}\n"
        state_str += f"  Timestep: {self.Tstep}\n"
        state_str += f"  Cell Coordinates: {self.cellcoord}\n"
        return state_str

class StateMachineWithConstraints(StateMachine):
    def __init__(self, name):
        super().__init__(name)
        self.transitions = []
        self._current_state = None

         # New metadata for yielding norm
        self.is_yielding_agent = False
        self.yield_obligation = None

    def add_state(self, state, initial=False):
        if not isinstance(state, ConstrainedState):
            state = ConstrainedState(state)
        super().add_state(state, initial)
        if initial:
            self._current_state = state  # Set initial state as current state
        return state

    def reset(self):
        # Reset both pysm's active state and the custom state reference.
        self.state = self.initial_state
        self._current_state = self.initial_state
        
    def transition(self, state):
        if isinstance(state, str):
            state = self.get_state(state)
        if state is None:
            raise ValueError("Invalid state.")
        
        # Check if the transition is valid
        current_state_name = self._current_state.name
        for transition in self.transitions:
            if transition['source'] == current_state_name and transition['target'] == state.name:
                # Transition is valid
                # Here you could add code to trigger exit actions, transition actions, and entry actions
                self._current_state = state
                return
        
        # If we get here, no valid transition was found
        raise ValueError(f"No valid transition from {current_state_name} to {state.name}")
    
    def get_state(self, state_name): 
        for state in self.states:
            st = str(state)
            if state.name == state_name:
                return state
        return None

    def dispatch_event(self, event_name):
        """Dispatch a semantic runtime event through the pysm state machine."""
        before = self.state
        self.dispatch(Event(event_name))
        self._current_state = self.state

        print(
            f"CCSM event '{event_name}': "
            f"{before.name} -> {self.state.name}"
        )

        return before is not self.state
    
def generate_path(action_list,start_coord,end_coord):
    state_matrix = [[i + j for j in range(10)] for i in range(0, 100, 10)]
    actions_mapping = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0), 4: (0, 0)}
    current_coord = start_coord
    path = [current_coord]
    for action in action_list:
        action_delta = actions_mapping[action]
        new_coord = (current_coord[0] + action_delta[0], current_coord[1] + action_delta[1])
        # new coordinates are within bounds
        if 0 <= new_coord[0] < len(state_matrix) and 0 <= new_coord[1] < len(state_matrix[0]):
            current_coord = new_coord
            path.append(current_coord)
    # print("List of Coordinates Reached:\n", path)
    return path
def plot_cell(ax, row, col, A, path, path_state_props):
    
    cell_width = 1.0
    cell_height = 1.0
    arrow_size = 0.2 

    if (row,col) == path[-1]:
        # Display the center value in blue if it's in the path
        ax.text(col + cell_width / 2, row + cell_height / 2, f"{(A[row, col]):.2f}", fontsize=8, ha='center',
                color='blue',fontweight='normal')
        
        # Shade the cell if it's in the path
        if (row, col) in path:
            rectangle = patches.Rectangle((col, row), cell_width, cell_height, linewidth=1, edgecolor='black',
                                        facecolor='yellow', alpha=0.5)
            ax.add_patch(rectangle)

        return
    neighbors = [
        (row - 1, col),  # Top
        (row + 1, col),  # Bottom
        (row, col - 1),  # Left
        (row, col + 1),  # Right
    ]
    min_wv = min(neighbors)

    # Shade the cell if it's in the path
    if (row, col) in path:
        rectangle = patches.Rectangle((col, row), cell_width, cell_height, linewidth=1, edgecolor='black',
                                      facecolor='yellow', alpha=0.5)
        ax.add_patch(rectangle)

    # Draw transparent arrows along the cells in the path in the same order
    if (row, col) in path and path.index((row, col)) < len(path) - 1:
        next_row, next_col = path[path.index((row, col)) + 1]
        dx = next_col - col
        dy = next_row - row
        ax.arrow(col + cell_width / 2, row + cell_height / 2, dx * cell_width, dy * cell_height,
                 head_width=arrow_size, head_length=arrow_size, fc='green', ec='green', alpha=0.8)
def calculate_state_props(path, A):
    state_props = []
    for i, cell in enumerate(path):
        row, col = cell
        dwv = A[row + 1, col] if row < A.shape[0] - 1 else 0
        twv = A[row - 1, col] if row > 0 else 0
        lwv = A[row, col - 1] if col > 0 else 0
        rwv = A[row, col + 1] if col < A.shape[1] - 1 else 0

        min_wv = min(lwv, rwv, twv, dwv)

        options = []
        if lwv == min_wv:
            options.append((row, col - 1))
        if rwv == min_wv:
            options.append((row, col + 1))
        if twv == min_wv:
            options.append((row - 1, col))
        if dwv == min_wv:
            options.append((row + 1, col))

        # Calculate "choice" as the state next to the current cell in the path
        choice = path[i + 1] if i + 1 < len(path) else None

        state_props.append({"options": options, "choice": choice})
    return state_props

def draw_csm(sm, constraints, agent, current_states):
    # Create a directed graph using pydot
    graph = pydot.Dot(graph_type='digraph')

    # Keep track of already added edges
    added_edges = set()

    # Add nodes and edges
    for transition in sm.transitions:
        source_node = pydot.Node(str(transition['source']))
        target_node = pydot.Node(str(transition['target']))

        if transition['source'] != transition['target']:
            if transition['target'] == 'end':
                continue
            else:
                edge_key = (transition['source'], transition['target'])
                if edge_key not in added_edges:
                    edge = pydot.Edge(source_node, target_node, label=', '.join(transition['events']))
                    graph.add_edge(edge)
                    added_edges.add(edge_key)

    # Add message nodes for constraints
    for state_name, constraint_keys in constraints.items():
        state_node = pydot.Node(str(state_name))
        graph.add_node(state_node)

        message_contents = "\n".join(constraint_keys)

        # Create a single message node with concatenated contents
        message_node = pydot.Node(message_contents, shape='note', style='filled', fillcolor='lightyellow')
        graph.add_node(message_node)

        # Connect the state node to the single message node
        graph.add_edge(pydot.Edge(state_node, message_node, style='dashed', dir='none', constraint=False))

    # Add a black point at the beginning
    start_node = pydot.Node('Start', shape='point', fillcolor='black', width='0.1', height='0.1')
    graph.add_node(start_node)
    graph.add_edge(pydot.Edge(start_node, pydot.Node(str(sm.transitions[0]['source']))))

    end_node = pydot.Node(' ', shape='doublecircle', style='filled', fillcolor='black', width='0.1', fixedsize=True)
    graph.add_node(end_node)

    # Find the last node in the transitions
    last_transition = sm.transitions[-2]
    last_node = pydot.Node(str(last_transition['source']))
    graph.add_node(last_node)
    last_timestep = len(current_states)-1
    # Add an edge from the last node to the end node
    graph.add_edge(pydot.Edge(last_node, end_node, label=f"t{last_timestep}"))

    graph_file_path = f"state_machine_graph_{agent}.png"
    graph.write(graph_file_path, format='png')

def get_state_prop(csm_desc):
    state_prop = csm_desc
    previous_state_name = None
    # print("\n\n\n\n")
    # print(state_prop)
    for state, state_info in state_prop.items():
        options = state_info['options']
        choice = state_info['choice']

        if choice in options:
            if choice is None:
                state_name = 'end'

            elif previous_state_name == 'regret_con':
                state_name = 'noregret'

            elif len(options) > 1:
                state_name = 'noregret_con'

            else:
                state_name = 'noregret'

        else:
            if choice is None:
                state_name='end'
            else:
                state_name = 'regret_con'
            

        state_info['state_name'] = state_name
        previous_state_name = state_name

    # print("--State Properties--\n",state_prop)

    return state_prop
def get_constraints(state_prop):
    constraints = {}

    # print(state_prop)
    for i, (key, value) in enumerate(state_prop.items()):
        timestep = f't{i + 1}'
        value['timestep'] = timestep
        # print(key)
        # print(value)
        options = value.get('options', [])
        state_name = value['state_name']
        if len(options) > 1: #to check if agent is constrained to once choice
            # print(key,value)
            if value['choice']==None:
                # print("yes")
                break
            constraint_key = f"obl({value['choice']}, '{value['timestep']}')"
            constraints[state_name] = set()
            if state_name in constraints:
                constraints[state_name].add(constraint_key)
            else:
                constraints[state_name] = {constraint_key}
    # print(constraints)
    return constraints

class TransitionCondition:
    def __init__(self, c_state, n_state, constraints):
        self.c_state = c_state
        self.n_state = n_state
        self.constraints = constraints

    def __call__(self, from_state, event):
        return self.constraints_met()

    def constraints_met(self):
        if self.constraints:
            checkcons = False
            key = list(self.constraints.keys())[0]
            pattern = r'obl\((.*?), \''
            match = re.search(pattern, key)
            if match:
                result = match.group(1)
            if str(self.c_state.cellcoord) == result:
                print("constraint condition met")
                return True
            else:
                print("Constraints not met!! ")
                raise ValueError("Constraints not met!!.")
        else:
            return True

def CSM(csm_desc, agent , is_yielding_agent=False):
    # Define states
    # start = ConstrainedState('start')
    noregret = ConstrainedState('noregret')
    regret = ConstrainedState('regret')
    noregret_con = ConstrainedState('noregret_con')
    regret_con = ConstrainedState('regret_con')
    end = ConstrainedState('end')

    # Create a state machine
    sm = StateMachineWithConstraints('StateMachineWithConstraints')

    # sm.add_state(start, initial=True)
    sm.add_state(noregret)
    sm.add_state(regret)
    sm.add_state(noregret_con)
    sm.add_state(regret_con)
    sm.add_state(end)
    #   Create a state machine with constraints

    state_prop = get_state_prop(csm_desc) #get state properties
    constraints= get_constraints(state_prop) #find constraints , tstep and cellcooridnates
    

    for key, value in state_prop.items():
        state_name = value['state_name']
        state = sm.get_state(state_name)
        state.Tstep = value['timestep']
    
    transitions = []

    
    # print("\n*************************CONSTRAINTS*********************\n",constraints)
    # print("\n*************************Timesptep*********************\n",state.Tstep)
    # print("\n*************************Cellcoord*********************\n",state.cellcoord)
    for state_name, constraint_keys in constraints.items():
        state = sm.get_state(state_name)
        state.constraints = {**state.constraints, **{k: state_name for k in constraint_keys}}
    cons = state.constraints
    if cons:
        key = list(cons.keys())[0]
        pattern = r'obl\((.*?), \''
        # print("\nkey", key)
        # print("\npattern",pattern)
        match = re.search(pattern, key)
        if match:
            result = match.group(1)
            statecellcoord =  result


    current_states = [state_info['state_name'] for state_info in state_prop.values()]
    
    # print(current_states)
    # print(state_prop)
    first_key = next(iter(state_prop.keys()))
    initial_state_name = state_prop[first_key]['state_name']
    # print(initial_state_name)
    initial_state = sm.get_state(initial_state_name)
    sm.add_state(initial_state, initial=True)
    added_transitions = set()  # Keep track of already added transitions
    
    for i in range(len(current_states) - 1):
        current_state = current_states[i]
        next_state = current_states[i + 1]
        event = f't{i + 1}'
        
        # Check if the transition has already been added
        if (current_state, next_state, event) not in added_transitions:
            c_state = sm.get_state(current_state)
            n_state = sm.get_state(next_state)
            transition_condition = TransitionCondition(c_state, n_state, c_state.constraints)
            sm.add_transition(c_state, n_state, events=[event], condition=transition_condition)
            sm.transitions.append({'source': current_state, 'target': next_state, 'events': [event]})
            added_transitions.add((current_state, next_state, event))  # Add the transition to the set
    
    final_state = current_states[-1]
    fnal_state = sm.get_state(final_state)
    end_state = sm.get_state('end')
    # sm.add_transition(fnal_state,end_state,events=[f't{len(transitions)}'])
    last_timestep = len(current_states)-1
    sm.add_transition(fnal_state, end_state, events=[f't{last_timestep}'])
    sm.transitions.append({
        'source': fnal_state.name,
        'target': end_state.name,
        'events': [f't{last_timestep}']
    })

    # print("\n---TRANSITION---\n",str(sm.transitions))
    # print(current_states)

    # NEW: decide whether this CSM belongs to the yielding role
    sm.is_yielding_agent = any(
        state_name in ["regret", "regret_con"]
        for state_name in current_states
    )

    # Keep track of the states that actually occur in this CSM
    sm.used_states = current_states


    # print("CSM yielding role:", sm.is_yielding_agent)

    draw_csm(sm, constraints, agent, current_states)
    return sm

def make_step_suboptimal_obligation(delta_R, contour_label="M"):
    return {
        "type": "Obl",
        "target": f"Contour({contour_label})",
        "duration": delta_R
    }



def make_clockwise_obligation(contour_label="L"):
    return {
        "type": "Obl",
        "target": f"Step Clockwise Contour({contour_label})",
        "duration": 1
    }

def get_ccsm_event(source_name, target_name):
    """
    Convert a concrete CSM transition into a semantic CCSM event.

    rel:
        The obligation attached to a constrained state has completed.

    goal_reached:
        The agent has reached its destination and may enter the end state.

    """
    if target_name == "end":
        return "goal_reached"

    if source_name in ("regret_con", "noregret_con"):
        return "rel"
    return None


def cross_scenario_CSM(smA, smB,delta_R_A,delta_R_B,contour_cells_A, contour_cells_B):
    cross_scenario_csmA = StateMachineWithConstraints('cross_scenario')
    cross_scenario_csmB = StateMachineWithConstraints('cross_scenario')

    # Copy role metadata from original CSMs
    cross_scenario_csmA.is_yielding_agent = smA.is_yielding_agent
    cross_scenario_csmB.is_yielding_agent = smB.is_yielding_agent

    # Store used state names from original CSMs
    cross_scenario_csmA.used_states = getattr(smA, "used_states", [])
    cross_scenario_csmB.used_states = getattr(smB, "used_states", [])

    
    # Add states
    for state in smA.states:
        cross_scenario_csmA.add_state(state)

    for state in smB.states:
        cross_scenario_csmB.add_state(state)

    # Add initial states
    initial_state_A = smA.get_state(smA.transitions[0]['source'])
    initial_state_B = smB.get_state(smB.transitions[0]['source'])

    cross_scenario_csmA.add_state(
        cross_scenario_csmA.get_state(initial_state_A.name),
        initial=True
    )

    cross_scenario_csmB.add_state(
        cross_scenario_csmB.get_state(initial_state_B.name),
        initial=True
    )

    # Preserve the transition structure, but replace concrete t1/t2/... labels
    # with semantic events that can be recognised at runtime.
    for transition in smA.transitions:
        source_state = cross_scenario_csmA.get_state(transition['source'])
        target_state = cross_scenario_csmA.get_state(transition['target'])
        event_name = get_ccsm_event(source_state.name, target_state.name)

        cross_scenario_csmA.add_transition(
            source_state,
            target_state,
            events=[event_name]
        )
        if event_name is None:
            continue

        cross_scenario_csmA.transitions.append({
            'source': source_state.name,
            'target': target_state.name,
            'events': [event_name]
        })

    for transition in smB.transitions:
        source_state = cross_scenario_csmB.get_state(transition['source'])
        target_state = cross_scenario_csmB.get_state(transition['target'])
        event_name = get_ccsm_event(source_state.name, target_state.name)

        cross_scenario_csmB.add_transition(
            source_state,
            target_state,
            events=[event_name]
        )

        cross_scenario_csmB.transitions.append({
            'source': source_state.name,
            'target': target_state.name,
            'events': [event_name]
        })

    # Remove concrete cell-level constraints
    for state in cross_scenario_csmA.states:
        state.constraints = {}

    for state in cross_scenario_csmB.states:
        state.constraints = {}

    #  Add abstract CCSM obligations
    for cross_csm, delta_R in [
    (cross_scenario_csmA, delta_R_A),
    (cross_scenario_csmB, delta_R_B)
]:


        if cross_csm.is_yielding_agent:
            cross_csm.norm_type = "yielding"

            # regret_con = first suboptimal/deviation step
            if "regret_con" in cross_csm.used_states:
                regret_state = cross_csm.get_state("regret_con")
                if regret_state:
                    regret_state.constraints = {
                        "Obl": make_step_suboptimal_obligation(delta_R)
                    }

            

        else:
            cross_csm.norm_type = "route_separation"

            # regret-free constrained case: take clockwise first step
            if "noregret_con" in cross_csm.used_states:
                noregret_state = cross_csm.get_state("noregret_con")
                if noregret_state:
                    noregret_state.constraints = {
                        "Obl": make_clockwise_obligation("L")
                    }

    with open("cross_scenario_csmA.pkl", "wb") as file:
        pickle.dump(cross_scenario_csmA, file)

    with open("cross_scenario_csmB.pkl", "wb") as file:
        pickle.dump(cross_scenario_csmB, file)

    return cross_scenario_csmA, cross_scenario_csmB

def initial_func(cellvalue, action_list, start_coord, end_coord, agent, delta_R=None):
    path = generate_path(action_list, start_coord, end_coord)
    path_state_props = [None] * len(path)
    path_state_props = calculate_state_props(path, cellvalue)
    csm_desc = {}

    for cell, state_prop in zip(path, path_state_props):
        csm_desc[cell] = {"options": state_prop['options'], "choice": state_prop['choice']}

    sm = CSM(csm_desc, agent)

    # Save the state machine as a pickle file
    with open(f"state_machine_{agent}.pkl", "wb") as file:
        pickle.dump(sm, file)

    return sm

def draw_cross_csm(cross_scenario_csms, agents, sm_list):
    # Create a directed graph using pydot
    graphs = []
    for i, csm in enumerate(cross_scenario_csms):
        graph = pydot.Dot(graph_type='digraph')

        graphs.append(graph)

        # Keep track of already added edges
        added_edges = set()

        # print(csm.transitions)
        # Add nodes and edges
        for transition in csm.transitions:
            source_node = pydot.Node(str(transition['source']))
            target_node = pydot.Node(str(transition['target']))

            if transition['source'] != transition['target']:
                if transition['target'] == 'end':
                    continue
                else:
                    edge_key = (transition['source'], transition['target'])
                    if edge_key not in added_edges:
                        edge = pydot.Edge(source_node, target_node, label=', '.join(transition['events']))
                        graph.add_edge(edge)
                        added_edges.add(edge_key)

        # Add message nodes for constraints
        for state in csm.states:
            if state.constraints:
                state_node = pydot.Node(str(state.name))
                graph.add_node(state_node)

                message_contents = "\n".join(
                    f"{key}: {value}"
                    for key, value in state.constraints.items()
                    )

                
                message_node = pydot.Node(f"constraint_{i}_{state.name}",label=message_contents,shape="note",style="filled",fillcolor="lightyellow")
                graph.add_node(message_node)

                # Connect the state node to the single message node
                graph.add_edge(pydot.Edge(state_node, message_node, style='dashed', dir='none', constraint=False))

        # print(graph.to_string())

        start_node = pydot.Node('Start', shape='point', fillcolor='black', width='0.1', height='0.1')
        graph.add_node(start_node)
        graph.add_edge(pydot.Edge(start_node, pydot.Node(str(csm.transitions[0]['source']))))

        end_node = pydot.Node(' ', shape='doublecircle', style='filled', fillcolor='black', width='0.1', fixedsize=True)
        graph.add_node(end_node)

        # print(graph.to_string())

        # Find the last node in the transitions
        last_transition = csm.transitions[-2]
        last_node = pydot.Node(str(last_transition['source']))
        graph.add_node(last_node)
        graph.add_edge(pydot.Edge(last_node, end_node))

        print("\n\n\n", graph.to_string())
        graph_file_path = f"cross_scenario_graph_{agents[i]}.png"
        graph.write(graph_file_path, format='png')

def display_ccsm():
    graph_file_path1 = "state_machine_graph_A.png"
    graph_file_path2 = "state_machine_graph_B.png"

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    plt.tight_layout()
    plt.show()
    plt.close()


    graph_file_path1 = "cross_scenario_graph_A.png"
    graph_file_path2 = "cross_scenario_graph_B.png"

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))

    plt.tight_layout()
    # plt.show()
    plt.close()

def roc_to_csm(cellvalue_A,action_list_A,start_coord_A,end_coord_A,cellvalue_B,action_list_B,start_coord_B,end_coord_B,
    delta_R_A, delta_R_B,contour_cells_A, contour_cells_B):
    print(delta_R_A, delta_R_B)


    print("\n========== CONTOURS PASSED TO CCSM ==========")

    print("\nAgent A contours:")
    for contour_label, cells in contour_cells_A.items():
        print(f"Contour {contour_label}:")
        print(sorted(cells))

    print("\nAgent B contours:")
    for contour_label, cells in contour_cells_B.items():
        print(f"Contour {contour_label}:")
        print(sorted(cells))

    print("=============================================\n")


    #CSM
    smA = initial_func(cellvalue_A,action_list_A,start_coord_A,end_coord_A,'A',delta_R=delta_R_A)
    smB = initial_func(cellvalue_B,action_list_B,start_coord_B,end_coord_B,'B',delta_R=delta_R_B)
    smA.initialize()
    smB.initialize()

    #CROSS CSM
    cross_scenario_csmA, cross_scenario_csmB = cross_scenario_CSM(smA, smB,delta_R_A,delta_R_B, contour_cells_A, contour_cells_B)
    cross_scenario_csms = [cross_scenario_csmA, cross_scenario_csmB]
    with open("cross_scenario_csms.pkl", "wb") as file:
        pickle.dump(cross_scenario_csms, file)

    agents = ['A', 'B']

    draw_cross_csm(cross_scenario_csms, agents, [smA, smB])
    display_ccsm()
    cross_scenario_csmA.initialize()
    cross_scenario_csmB.initialize()
    return (cross_scenario_csmA, cross_scenario_csmB)

