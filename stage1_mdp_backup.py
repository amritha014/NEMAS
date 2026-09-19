import collections
import csv
import json
import math
import os
import pickle
import random
import time
import uuid
from itertools import chain

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation

import Agentgrid as agentgrid
import mdp as md
import mdp_gym.env as en
import mdp_gym.utils_gym as ut
import mdptoolbox as mdptoolbox
from mdp import MDP
from stage1_contour import gen_contour
from stage1_isomorphism import compare_contour_trees
from stage1_regcalc import reg_calc, threshold_reg
from stage2_roc_to_csm import roc_to_csm
from stage3_csm_to_roc import csm_to_roc
from stage3_negotiation import Coordinator

# NE_agent = 0  # HARDCODED agents_current_pos[0] 0 is considered here as NE
# other_agent = 1  # HARDCODED agents_current_pos[0] 0 is considered here as NE

ADJ_CELL_COUNT = 5
ntive_inf = -math.inf
right_action = 0
top_action = 1
left_action = 2
bottom_action = 3
stay_action = 4
world_data = {}
agents = set()
agent_objects={}
agents_for_ROC_copy = set()
agents_individual_reward = set()
agents_count = []
world_params = {}
optimal_policy = {}
cell_values = {}
agent_mdp = {}
agent_mdp_individual_reward = {}
agent_step = {}
current_agent_step_tracking = {}
initial_state = {}
agent_action = {}
agent_planaction = {}
agent_initial_action = {}
agent_action_individual_reward = {}
current_state = {}
new_state = {}
finished = False
goal = {}
agent_gridworld = {}
agent_gridworld_individual_reward = {}
accumulated_reward = {}
accumulated_reward_individual = {}
agent_reward = {}
value = {}
gamma_val = {}
cost = {}
agent_gamma = {}
agent_gamma_dict = {}
agent_gamma_individual_reward = {}
agent_gamma_individual_reward_dict = {}
agents_param = {}
action_mapping = {
    "right": 0,
    "up": 1,
    "left": 2,
    "down": 3,
    "stay": 4
}
agent_section = {}
temp_agent_state = {}
pairs = []
agents_current_pos = {}
simulate_actions={}

np.set_printoptions(linewidth=200)
np.set_printoptions(threshold=np.inf)
def get_reward(
    startRow, startCol, goalRow, goalCol, oCol, oRow, num_states, m, optimal_policy, rm
):

    cur_row = startRow
    cur_col = startCol

    opt_row = [startRow]
    opt_col = [startCol]
    max_points = num_states
    cur_point = 0
    # print("************************************")
    # print('current state  row: {}, col: {}'.format(self.startRow, self.startCol))
    total_reward = 0
    while 1:

        cur_state = int(m[0][cur_row][cur_col])
        cur_opt_action = optimal_policy[str(cur_state)]
        total_reward += rm[cur_state][cur_opt_action]

        if cur_opt_action == 0:
            cur_row = cur_row
            cur_col = cur_col + 1
        elif cur_opt_action == 1:
            cur_row = cur_row - 1
            cur_col = cur_col
        elif cur_opt_action == 2:
            cur_row = cur_row
            cur_col = cur_col - 1
        else:
            cur_row = cur_row + 1
            cur_col = cur_col

        opt_row.append(cur_row)
        opt_col.append(cur_col)

        cur_point += 1

        if cur_row == goalRow:
            if cur_col == goalCol:
                print("Goal Reached!!")
                break

        if cur_point == max_points:
            print("Steps limit over!!")
            break

    return total_reward

def visualize_path(
    startRow,
    startCol,
    goalRow,
    goalCol,
    oCol,
    oRow,
    num_states,
    m,
    optimal_policy,
    rm,
    algorithm,
):

    # Visualize path
    cur_row = startRow
    cur_col = startCol

    opt_row = [startRow]
    opt_col = [startCol]
    max_points = num_states
    cur_point = 0
    agent_policy = []
    # print('current state  row: {}, col: {}'.format(self.startRow, self.startCol))
    while 1:

        cur_state = int(m[0][cur_row][cur_col])
        cur_opt_action = optimal_policy[str(cur_state)]

        agent_policy.append((cur_state, cur_opt_action))
        # print(agent_policy)

        if cur_opt_action == 0:
            cur_row = cur_row
            cur_col = cur_col + 1
        elif cur_opt_action == 1:
            cur_row = cur_row - 1
            cur_col = cur_col
        elif cur_opt_action == 2:
            cur_row = cur_row
            cur_col = cur_col - 1
        else:
            cur_row = cur_row + 1
            cur_col = cur_col

        opt_row.append(cur_row)
        opt_col.append(cur_col)
        cur_point += 1

        if cur_row == goalRow:
            if cur_col == goalCol:
                print("Goal Reached!!")
                return agent_policy
                break

        if cur_point == max_points:
            print("Steps limit over!!")
            break

def visualize_policy(
    maxRow,
    maxCol,
    startRow,
    startCol,
    goalRow,
    goalCol,
    oCol,
    oRow,
    num_states,
    m,
    optimal_policy,
    rm,
    algorithm,
    agent,
):

    if agent == 0:
        # Visualize world
        fig, ax = plt.subplots(figsize=(5.8, 5.8))
        plt.ion()
        ax.scatter(oCol, oRow, marker="s", s=700, c="black")
        ax.scatter(startCol, startRow, s=700, c="b")
        ax.scatter(goalCol, goalRow, s=700, c="g")
        plt.axis("equal")
        plt.axis("tight")

        # Arrow config
        arrow_head_len = 0.1
        len_arrow = 1 - 2 * arrow_head_len
        arrow = {}
        arrow["right"] = {"sx": -1 * (len_arrow / 2), "sy": 0, "dx": len_arrow, "dy": 0}
        arrow["top"] = {"sx": 0, "sy": (len_arrow / 2), "dx": 0, "dy": -1 * len_arrow}
        arrow["left"] = {"sx": (len_arrow / 2), "sy": 0, "dx": -1 * len_arrow, "dy": 0}
        arrow["bottom"] = {
            "sx": 0,
            "sy": -1 * (len_arrow / 2),
            "dx": 0,
            "dy": len_arrow,
        }

        # Visualize path
        cur_row = startRow
        cur_col = startCol

        opt_row = [startRow]
        opt_col = [startCol]
        max_points = num_states
        cur_point = 0

        # print('current state  row: {}, col: {}'.format(self.startRow, self.startCol))
        for cur_row in range(maxRow):
            for cur_col in range(maxCol):

                cur_state = int(m[0][cur_row][cur_col])
                cur_opt_action = optimal_policy[str(cur_state)]

                if cur_opt_action == 0:
                    direction = "right"
                    x = arrow[direction]["sx"] + cur_col
                    y = arrow[direction]["sy"] + cur_row
                    dx = arrow[direction]["dx"]
                    dy = arrow[direction]["dy"]

                elif cur_opt_action == 1:
                    direction = "top"
                    x = arrow[direction]["sx"] + cur_col
                    y = arrow[direction]["sy"] + cur_row
                    dx = arrow[direction]["dx"]
                    dy = arrow[direction]["dy"]

                elif cur_opt_action == 2:
                    direction = "left"
                    x = arrow[direction]["sx"] + cur_col
                    y = arrow[direction]["sy"] + cur_row
                    dx = arrow[direction]["dx"]
                    dy = arrow[direction]["dy"]

                else:
                    direction = "bottom"
                    x = arrow[direction]["sx"] + cur_col
                    y = arrow[direction]["sy"] + cur_row
                    dx = arrow[direction]["dx"]
                    dy = arrow[direction]["dy"]

                ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.1, fc="k", ec="k")

        figname = "policy_0_" + algorithm
        fig.savefig(figname)
        plt.close()

    elif agent == 1:
        # Visualize world
        fig, ax = plt.subplots(figsize=(5.8, 5.8))
        plt.ion()
        ax.scatter(oCol, oRow, marker="s", s=700, c="black")
        ax.scatter(startCol, startRow, s=700, c="b")
        ax.scatter(goalCol, goalRow, s=700, c="g")
        plt.axis("equal")
        plt.axis("tight")

        # Arrow config
        arrow_head_len = 0.1
        len_arrow = 1 - 2 * arrow_head_len
        arrow = {}
        arrow["right"] = {"sx": -1 * (len_arrow / 2), "sy": 0, "dx": len_arrow, "dy": 0}
        arrow["top"] = {"sx": 0, "sy": (len_arrow / 2), "dx": 0, "dy": -1 * len_arrow}
        arrow["left"] = {"sx": (len_arrow / 2), "sy": 0, "dx": -1 * len_arrow, "dy": 0}
        arrow["bottom"] = {
            "sx": 0,
            "sy": -1 * (len_arrow / 2),
            "dx": 0,
            "dy": len_arrow,
        }

        # Visualize path
        cur_row = startRow
        cur_col = startCol

        opt_row = [startRow]
        opt_col = [startCol]
        max_points = num_states
        cur_point = 0

        # print('current state  row: {}, col: {}'.format(self.startRow, self.startCol))
        for cur_row in range(maxRow):
            for cur_col in range(maxCol):

                cur_state = int(m[0][cur_row][cur_col])
                cur_opt_action = optimal_policy[str(cur_state)]

                if cur_opt_action == 0:
                    direction = "right"
                    x = arrow[direction]["sx"] + cur_col
                    y = arrow[direction]["sy"] + cur_row
                    dx = arrow[direction]["dx"]
                    dy = arrow[direction]["dy"]

                elif cur_opt_action == 1:
                    direction = "top"
                    x = arrow[direction]["sx"] + cur_col
                    y = arrow[direction]["sy"] + cur_row
                    dx = arrow[direction]["dx"]
                    dy = arrow[direction]["dy"]

                elif cur_opt_action == 2:
                    direction = "left"
                    x = arrow[direction]["sx"] + cur_col
                    y = arrow[direction]["sy"] + cur_row
                    dx = arrow[direction]["dx"]
                    dy = arrow[direction]["dy"]

                else:
                    direction = "bottom"
                    x = arrow[direction]["sx"] + cur_col
                    y = arrow[direction]["sy"] + cur_row
                    dx = arrow[direction]["dx"]
                    dy = arrow[direction]["dy"]

                ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.1, fc="k", ec="k")

        figname = "policy_1_" + algorithm
        fig.savefig(figname)
        plt.close()

def fit_policy(st, rm, gamma, num_states):
    """
    This function trains an optimal policy using Markov Decision Process using MDPToolbox
    using PolicyIteration

    """
    iterations = list(range(1, 100, 10))
    data_policy = {}
    data_policy["convergence"] = {}

    for iter in iterations:

        # print('Current Iteration: {}'.format(iter))

        data_policy[str(iter)] = {}

        tot_time_start = time.time()

        vi = mdptoolbox.mdp.PolicyIteration(st, rm, gamma, max_iter=100, eval_type=1)
        # vi.setVerbose()
        time_iter, iter_value, iter_policy, policy_change, policies = vi.run(
            max_iter=iter
        )
        tot_time_end = time.time()
        tot_time = tot_time_end - tot_time_start

        policy_change = [int(x) for x in policy_change]
        if np.any(np.array(iter_value) > iter):
            raise ValueError(
                "Value loop of Policy Iteration not stopping at maximum iterations provided"
            )

        data_policy[str(iter)]["tot_time"] = tot_time
        data_policy[str(iter)]["time_iter"] = time_iter
        data_policy[str(iter)]["policy_iter"] = iter_policy
        data_policy[str(iter)]["value_iter"] = iter_value
        data_policy[str(iter)]["policy_change"] = policy_change

    # print('Convergence')
    tot_time_start = time.time()
    vi = mdptoolbox.mdp.PolicyIteration(st, rm, gamma, max_iter=100, eval_type=1)
    time_iter, iter_value, iter_policy_policy, policy_change, policies = vi.run(
        max_iter=100
    )
    tot_time_end = time.time()

    policy_change = [int(x) for x in policy_change]
    policies = [tuple(int(x) for x in opt_policy) for opt_policy in policies]
    optimal_policy = vi.policy
    expected_values = vi.V
    optimal_policy = tuple(int(x) for x in optimal_policy)
    expected_values = tuple(float(x) for x in expected_values)

    optimal_policy = dict(zip(list(range(num_states)), list(optimal_policy)))
    expected_values = list(expected_values)
    cell_values = expected_values
    # print(cell_values)
    policies = [
        dict(zip(list(range(num_states)), list(opt_policy))) for opt_policy in policies
    ]

    data_policy["convergence"]["tot_time"] = tot_time_end - tot_time_start
    data_policy["convergence"]["time_iter"] = time_iter
    data_policy["convergence"]["policy_iter"] = iter_policy_policy
    data_policy["convergence"]["value_iter"] = iter_value
    data_policy["convergence"]["policy_change"] = policy_change
    data_policy["convergence"]["optimal_policy"] = optimal_policy
    data_policy["convergence"]["expected_values"] = expected_values
    data_policy["convergence"]["policies"] = policies

    return data_policy, optimal_policy, cell_values

def store_to_file(data, file_name):

    with open(file_name, "w") as outfile:
        json.dump(data, outfile)

def fit_value(st, rm, gamma, num_states):
    """
    This function trains an optimal policy using Markov Decision Process using MDPToolbox
    using ValueIteration

    """
    iterations = list(range(1, 100, 10))
    data_value = {}
    data_value["convergence"] = {}
    for iter in iterations:

        # print('Current Iteration: {}'.format(iter))
        data_value[str(iter)] = {}

        tot_time_start = time.time()
        vi = mdptoolbox.mdp.ValueIteration(st, rm, gamma, max_iter=100, epsilon=0.0001)
        # vi.setVerbose()
        time_iter, iter_value, variation, policies = vi.run(max_iter=iter)
        tot_time_end = time.time()
        tot_time = tot_time_end - tot_time_start

        if iter_value > iter:
            raise ValueError("ValueIteration is not stopping at maximum iterations")

        data_value[str(iter)]["tot_time"] = tot_time
        data_value[str(iter)]["time_iter"] = time_iter
        data_value[str(iter)]["value_iter"] = iter_value
        data_value[str(iter)]["variation"] = variation

    # print('Convergence')
    tot_time_start = time.time()
    vi = mdptoolbox.mdp.ValueIteration(st, rm, gamma, max_iter=100, epsilon=0.0001)
    time_iter, iter_value, variation, policies = vi.run(max_iter=100)
    tot_time_end = time.time()

    optimal_policy = vi.policy
    expected_values = vi.V
    policies = [tuple(int(x) for x in opt_policy) for opt_policy in policies]
    optimal_policy = tuple(int(x) for x in optimal_policy)
    expected_values = tuple(float(x) for x in expected_values)

    optimal_policy = dict(zip(list(range(num_states)), list(optimal_policy)))
    expected_values = list(expected_values)
    policies = [
        dict(zip(list(range(num_states)), list(opt_policy))) for opt_policy in policies
    ]

    data_value["convergence"]["tot_time"] = tot_time_end - tot_time_start
    data_value["convergence"]["time_iter"] = time_iter
    data_value["convergence"]["value_iter"] = iter_value
    data_value["convergence"]["variation"] = variation
    data_value["convergence"]["optimal_policy"] = optimal_policy
    data_value["convergence"]["expected_values"] = expected_values
    data_value["convergence"]["policies"] = policies

    # print(expected_values)

    return data_value

def plot_analysis(file_data_world, file_data_policy, agent):

    iterations = list(range(1, 1000, 10))
    with open(file_data_world) as json_data:
        data_world = json.load(json_data)
    with open(file_data_policy) as json_data:
        data_policy = json.load(json_data)


    # Visualize path and policy for policy iteration and value iteration
    visualize_path(
        data_world["startRow"],
        data_world["startCol"],
        data_world["goalRow"],
        data_world["goalCol"],
        data_world["oCol"],
        data_world["oRow"],
        data_world["num_states"],
        data_world["m"],
        data_policy,
        data_world["rm"],
        "policy_iteration",
    )
    # visualize_path(data_world['startRow'], data_world['startCol'], data_world['goalRow'], data_world['goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'], data_value['convergence']['optimal_policy'], data_world['rm'],'value_iteration')
    visualize_policy(
        data_world["maxRow"],
        data_world["maxCol"],
        data_world["startRow"],
        data_world["startCol"],
        data_world["goalRow"],
        data_world["goalCol"],
        data_world["oCol"],
        data_world["oRow"],
        data_world["num_states"],
        data_world["m"],
        data_policy,
        data_world["rm"],
        "policy_iteration",
        agent,
    )

class GridWorld:
    def __init__(self, max_row, max_col, wall_rows, wall_cols):
        self.max_row = max_row
        self.max_col = max_col
        self.grid = np.zeros((max_row, max_col))
        for r, c in zip(wall_rows, wall_cols):
            self.grid[r, c] = 1  # 1 represents a wall

    def is_valid_move(self, row, col):
        return 0 <= row < self.max_row and 0 <= col < self.max_col and self.grid[row, col] == 0

class Agent:
    next_tree_id = 1000
    def __init__(self, id, start_row=None, start_col=None, goal_row=None, goal_col=None, color=None):
        self.id = id
        self.row = start_row
        self.col = start_col
        self.goal_row = goal_row
        self.goal_col = goal_col
        self.color = color
        self.contour_trees = {}
        self.constrained_state_machines = {}
        self.csm_usage_count = {}
        self.csm_passed_by = {}

    def update_attributes(self, start_row, start_col, goal_row, goal_col, color):
        self.row = start_row
        self.col = start_col
        self.goal_row = goal_row
        self.goal_col = goal_col
        self.color = color

   
    def add_contour_tree(self, contour_tree,id):
        if id:
            tree_id = id
            self.contour_trees[tree_id] = contour_tree
        else:
            tree_id = f"tree_{self.id}_{random.randint(1000, 9999)}"
            self.contour_trees[tree_id] = contour_tree
        return tree_id

    def add_constrained_state_machines(self, tree_id, constrained_state_machines):
        if tree_id in self.contour_trees:
            self.constrained_state_machines[tree_id] = constrained_state_machines
            self.csm_usage_count[tree_id] = 1  # Initialize usage count
        else:
            raise ValueError(f"No contour tree found with ID {tree_id}")

    def increment_csm_usage(self, tree_id):
        if tree_id in self.csm_usage_count:
            self.csm_usage_count[tree_id] += 1
        elif tree_id in self.constrained_state_machines:
            self.csm_usage_count[tree_id] = 1
        else:
            print(f"Warning: No CSM found with tree ID {tree_id}. Initializing count.")
            self.csm_usage_count[tree_id] = 1

    def get_csm_usage_count(self, tree_id):
        return self.csm_usage_count.get(tree_id, 0)

    def get_all_contour_trees(self):
        return self.contour_trees

    def get_constrained_state_machines(self, tree_id):
        print(self)
        return self.constrained_state_machines.get(tree_id)

    def move(self, direction):
        if direction == 0:  # right
            self.col += 1
        elif direction == 1:  # up
            self.row -= 1
        elif direction == 2:  # left
            self.col -= 1
        elif direction == 3:  # down
            self.row += 1

    def at_goal(self):
        return self.row == self.goal_row and self.col == self.goal_col

class AgentState:
    def __init__(
        self,
        maxRow,
        maxCol,
        num_obstacle_pts,
        cor_pr,
        wr_pr,
        n_actions,
        startRow,
        startCol,
        goalRow,
        goalCol,
        stayReward,
        goalReward,
        obstReward,
        moveReward,
        gamma,
    ):

        self.agents = agents
        self.maxRow = maxRow
        self.maxCol = maxCol
        self.num_obstacle_pts = num_obstacle_pts
        self.cor_pr = cor_pr
        self.wr_pr = wr_pr
        self.n_actions = n_actions
        self.startRow = startRow
        self.startCol = startCol
        self.goalRow = goalRow
        self.goalCol = goalCol
        self.stayReward = stayReward
        self.goalReward = goalReward
        self.obstReward = obstReward
        self.moveReward = moveReward
        self.gamma = gamma
        self.n_agents = 4

        # Initializing model outputs
        self.not_occupied = 0
        self.oRow = []
        self.oCol = []
        self.m = np.zeros((2, self.maxRow, self.maxCol), dtype=int)
        # self.state_matrix = np.zeros((self.maxRow, self.maxCol))
        self.st = np.zeros(
            (self.n_actions, self.maxRow * self.maxCol, self.maxRow * self.maxCol)
        )
        self.num_states = self.maxRow * self.maxCol
        self.rm = np.ones((self.num_states, self.n_actions)) * self.moveReward

    def rotate(self, l, n):

        """
        rotate a list n times in anticlockwise direction
        """
        return l[n:] + l[:n]

    def get_obstacles(self):
        """
        returns a list of obstacle state indexes for rows and columns

        """
        wallRows = [7,8,8,7,8,9,10,11,12,13,0,2,3,5,7,13,0,2,3,5,10,11,7,13,7,8,9,10,11,12,13,0,2,3,5,7,8,9,10,11,12,13,14,15,0,2,3,5,7,15,0,2,3,5,7,9,10,11,13,15,7,9,13,15,7,9,11,12,13,15,7,15,7,8,9,11,12,13,14,15,]
        wallCols = [0,4,2,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,10,10,10,10,11,11,11,11,11,11,12,12,13,13,13,13,13,13,13,13,]
        	
        self.oRow.extend(wallRows)
        self.oCol.extend(wallCols)
        # print(self.oRow)
        # print(self.oCol)
        return None

    def get_world(self):
        return self.m

    def get_ROC_matrix(
        self, NE_cur_state, colstartidx, colendidx, rowstartidx, rowendidx
    ):
        # state_matrix = AgentState.get_state_matrix()
        # print(self.maxRow)
        s_matrix = np.arange(self.maxRow * self.maxCol)
        # print(s_matrix)
        state_matrix = s_matrix.reshape(self.maxRow, self.maxCol)
        # print(state_matrix)
       
        # print(state_matrix)
        submatrix_temp = [
            x[colstartidx:colendidx] for x in state_matrix[rowstartidx:rowendidx]
        ]

        # print("NE AGENT's STATE is:", NE_cur_state)

        submatrix_temp_list = []
        for array in submatrix_temp:
            submatrix_temp_list.append(array.tolist())
        # print(submatrix_temp_list)
        submatrix = np.array(submatrix_temp_list)
        submatrix.reshape(len(submatrix_temp_list[0]), len(submatrix_temp_list))
        # submatrix.reshape((ADJ_CELL_COUNT*2)+1,(ADJ_CELL_COUNT*2)+1)
        # print(submatrix)
        return state_matrix, submatrix, NE_cur_state, colstartidx, colendidx, rowstartidx, rowendidx

    def get_grid_world_ROC(grid_world, colstartidx, colendidx, rowstartidx, rowendidx):

        state_matrix = grid_world
        submatrix_temp = [
            x[colstartidx:colendidx] for x in state_matrix[rowstartidx:rowendidx]
        ]
        submatrix_temp_list = []
        for array in submatrix_temp:
            submatrix_temp_list.append(array.tolist())
        # print(submatrix_temp_list)
        submatrix = np.array(submatrix_temp_list)
        submatrix.reshape(len(submatrix_temp_list[0]), len(submatrix_temp_list))
        # submatrix.reshape((ADJ_CELL_COUNT*2)+1,(ADJ_CELL_COUNT*2)+1)
        # print(submatrix)
        return submatrix

    def get_ROC_cell_value_matrix(
        self, agent, NE_cur_state, colstartidx, colendidx, rowstartidx, rowendidx
    ):

        # state_matrix = AgentState.get_state_matrix()
        for i in agent_gridworld:
            if i == agent:
                Valuematrix = np.array(cell_values[i])

        Valuematrix_shape = Valuematrix.reshape(self.maxRow, self.maxCol)
        submatrix_temp = [
            x[colstartidx:colendidx] for x in Valuematrix_shape[rowstartidx:rowendidx]
        ]

        # print("NE AGENT's STATE is:", NE_cur_state)

        submatrix_temp_list = []
        for array in submatrix_temp:
            submatrix_temp_list.append(array.tolist())
        # print(submatrix_temp_list)
        submatrix = np.array(submatrix_temp_list)
        submatrix.reshape(len(submatrix_temp_list[0]), len(submatrix_temp_list))
        # submatrix.reshape((ADJ_CELL_COUNT*2)+1,(ADJ_CELL_COUNT*2)+1)
        # print(submatrix)
        return Valuematrix_shape, submatrix

    # TODO? passing of self not needed or pass NONE instead of passing agent_gridworld unnecessarily, directly we could take self.maxrow... change this later
    def get_coordinates_ofgiven_state(self, state_toget_coordinates):
        coordinates_ofgiven_state = []
        s_matrix = np.arange(self.maxRow * self.maxCol)
        # print(s_matrix)
        state_matrix = s_matrix.reshape(self.maxRow, self.maxCol)
        # print(state_matrix)
        coordinates = np.where(state_matrix == state_toget_coordinates)
        coordinates_ofgiven_state.append((coordinates[0][0], coordinates[1][0]))
        return coordinates_ofgiven_state


    def get_coordinates_nearto_givenstate(self, coordinates_ofgiven_state):
        coordinates_nearto_givenstate = []
        right_coord = []
        left_coord = []
        top_coord = []
        bottom_coord = []
        diagonal_nearby_coords = []

        current_x, current_y = coordinates_ofgiven_state[0]

        for i in range(1, ADJ_CELL_COUNT + 1):
            right_state = current_y + i
            if right_state < 16:
                right_coord.append((current_x, right_state))
                coordinates_nearto_givenstate.append((current_x, right_state))
            # else:
                # print("INVALID right_state for", coordinates_ofgiven_state, right_state)
        # print("right_states", right_coord)

        for i in range(1, ADJ_CELL_COUNT + 1):
            top_state = current_x - i
            if top_state >= 0:
                top_coord.append((top_state, current_y))
                coordinates_nearto_givenstate.append((top_state, current_y))
            # else:
                # print("INVALID top_state for", coordinates_ofgiven_state, top_state)
        # print("top_states", top_coord)

        for i in range(1, ADJ_CELL_COUNT + 1):
            left_state = current_y - i
            if left_state >= 0:
                left_coord.append((current_x, left_state))
                coordinates_nearto_givenstate.append((current_x, left_state))
            # else:
                # print("INVALID left_state for", coordinates_ofgiven_state, left_state)
        # print("left_states", left_coord)

        for i in range(1, ADJ_CELL_COUNT + 1):
            bottom_state = current_x + i
            if bottom_state < 16:
                bottom_coord.append((bottom_state, current_y))
                coordinates_nearto_givenstate.append((bottom_state, current_y))
            # else:
                # print("INVALID bottom_state for", coordinates_ofgiven_state, bottom_state)
        # print("bottom_states", bottom_coord)

        top_left = (current_x - 1, current_y - 1)
        if top_left[0] >= 0 and top_left[1] >= 0:
            diagonal_nearby_coords.append(top_left)

        top_right = (current_x - 1, current_y + 1)
        if top_right[0] >= 0 and top_right[1] < 16:
            diagonal_nearby_coords.append(top_right)

        bottom_left = (current_x + 1, current_y - 1)
        if bottom_left[0] < 16 and bottom_left[1] >= 0:
            diagonal_nearby_coords.append(bottom_left)

        bottom_right = (current_x + 1, current_y + 1)
        if bottom_right[0] < 16 and bottom_right[1] < 16:
            diagonal_nearby_coords.append(bottom_right)

        # print("coordinates_nearto_givenstate", coordinates_nearto_givenstate)

        colstartidx = min(y_pos for (_, y_pos) in left_coord) if left_coord else current_y
        # print("left_coord", left_coord)
        # print("colstartidx", colstartidx)

        colendidx = max(y_pos for (_, y_pos) in right_coord) + 1 if right_coord else current_y + 1
        # print("right_coord", right_coord)
        # print("colendidx", colendidx)

        rowstartidx = min(x_pos for (x_pos, _) in top_coord) if top_coord else current_x
        # print("top_coord", top_coord)
        # print("rowstartidx", rowstartidx)

        rowendidx = max(x_pos for (x_pos, _) in bottom_coord) + 1 if bottom_coord else current_x + 1
        # print("BOTTOM_COORD_", bottom_coord)
        # print("rowendidx", rowendidx)

        # print(colstartidx, colendidx, rowstartidx, rowendidx, diagonal_nearby_coords)
        return (
            coordinates_nearto_givenstate,
            colstartidx,
            colendidx,
            rowstartidx,
            rowendidx,
            diagonal_nearby_coords,
        )
       
    def get_state_trans_dict(self, state_matrix):
        state_trans_dict = {}
        s_matrix = np.arange(self.maxRow * self.maxCol)
        # print(s_matrix)
        state_matrix = s_matrix.reshape(self.maxRow, self.maxCol)
        # print(state_matrix)
        for array in state_matrix:
            for ele in array:
                state_trans_dict[ele] = AgentState.get_coordinates_ofgiven_state(
                    self, ele
                )

        return state_trans_dict

    def get_ROC_trans_dict(self, ROC):

        ROC_trans_dict = {}
        maxRow = len(ROC)
        maxCol = len(ROC[0])
        for array in ROC:
            for ele in array:
                ROC_trans_dict[ele] = AgentState.get_coordinates_ofgiven_ROC(
                    self, ROC, ele
                )
        # print(ROC)
        return ROC_trans_dict

    def get_coordinates_ofgiven_ROC(self, ROC, state_toget_coordinates):
        coordinates_ofgiven_ROC = []
        coordinates = np.where(ROC == state_toget_coordinates)
        coordinates_ofgiven_ROC.append((coordinates[0][0], coordinates[1][0]))
        return coordinates_ofgiven_ROC

    def get_statematrix_to_ROC_coordinate(
        ROC,
        state_matrix,
        ROC_trans_dict,
        state_trans_dict,
        NE_statematrix_coord,
        other_agent_statematrix_coord,
    ):

        for key, val in state_trans_dict.items():
            # print(val[0],NE_statematrix_coord[0])
            if val[0] == NE_statematrix_coord[0]:
                a_state = key
                # print(a_state)
                for key, val in ROC_trans_dict.items():
                    if key == a_state:
                        NE_ROCmatrix_coord = val[0]
            # print(val[0],other_agent_statematrix_coord[0])
            if val[0] == other_agent_statematrix_coord[0]:
                a_state = key
                # print(a_state)
                for key, val in ROC_trans_dict.items():
                    if key == a_state:
                        other_agent_ROCmatrix_coord = val[0]


        return NE_ROCmatrix_coord, other_agent_ROCmatrix_coord


    def get_combinedagent_matrix(
        state_matrix, ROC, grid_world, ROC_grid_world, NE_cur_state, other_agents_state
    ):

        
        for agent in agent_gridworld:
            # print("AGENTSSSSSSSSSSSSSSSSSSSSSSSSS", agents, agent_gridworld)
            if agent == current_agent_1:
                NE_statematrix_coord = AgentState.get_coordinates_ofgiven_state(
                    agent_gridworld[current_agent_1], NE_cur_state
                )
                # print(NE_statematrix_coord)
            if agent == current_agent_2:
                other_agent_statematrix_coord = (
                    AgentState.get_coordinates_ofgiven_state(
                        agent_gridworld[current_agent_2], other_agents_state[0]
                    )
                )


        for agent in agent_gridworld:
            if agent == current_agent_1:
                state_trans_dict = AgentState.get_state_trans_dict(
                    agent_gridworld[agent], state_matrix
                )
                ROC_trans_dict = AgentState.get_ROC_trans_dict(
                    agent_gridworld[agent], ROC
                )

        (
            NE_ROCmatrix_coord,
            other_agent_ROCmatrix_coord,
        ) = AgentState.get_statematrix_to_ROC_coordinate(
            ROC,
            state_matrix,
            ROC_trans_dict,
            state_trans_dict,
            NE_statematrix_coord,
            other_agent_statematrix_coord,
        )

        print(
            # "NE_ROCmatrix_coord,other_agent_ROCmatrix_coord",
            NE_ROCmatrix_coord,
            other_agent_ROCmatrix_coord,
        )
     
        ROC_wall_coords = []
        ROC_wall_coords_temp = np.where(ROC_grid_world == 1)
        ROC_wall_coords = list(zip(ROC_wall_coords_temp[0], ROC_wall_coords_temp[1]))
        # print(ROC_wall_coords)

        ROC_list_matrix = ROC.tolist()
        # print(ROC_list_matrix)
        for eachlist in ROC_list_matrix:
            ROC_list_matrix = np.where(eachlist, {"_"}, ROC_list_matrix)
            for i in ROC_wall_coords:
                ROC_list_matrix[i] = {"T"}
            if NE_ROCmatrix_coord:
                # print("ROC_list_matrix[NE_ROCmatrix_coord]",ROC_list_matrix[NE_ROCmatrix_coord])
                ROC_list_matrix[NE_ROCmatrix_coord] = {"A"}

            if other_agent_ROCmatrix_coord:
                # print("ROC_list_matrix[other_agent_ROCmatrix_coord]",ROC_list_matrix[other_agent_ROCmatrix_coord])
                ROC_list_matrix[other_agent_ROCmatrix_coord] = {"B"}

        return ROC_list_matrix

    def build_map(self):

        """
        builds the map using obstacle state indexes
        """
        # self.get_coordinates(44)
        cur_state = 0
        for row in range(self.maxRow):
            for col in range(self.maxCol):
                self.m[0][row][col] = cur_state
                cur_state += 1

        for row, col in zip(self.oRow, self.oCol):
            if self.not_occupied == 1:

                self.m[1][row][col] = 0
            else:
                self.m[1][row][col] = 1
                # commeting now
        # print(self.m)
        return None

    def build_st_trans_matrix(self, terminal_states): 
        """
        Function that builds state transition model for input to the MDP

        """
        if (self.cor_pr + (self.n_actions - 1) * self.wr_pr) < 0.99:
            raise ValueError("Sum of probabilities dont match")

        if self.not_occupied != 0:
            if self.not_occupied != 1:
                raise ValueError("not occupied should be either zero or one")

        # Actions should start from right and go in anti clockwise direction

        for row in range(self.maxRow):
            for col in range(self.maxCol):
                if row == terminal_states[0][0] and col == terminal_states[0][1]:

                    ter_state = self.m[0][row][col]

        act_map = []
        cur_states_a1 = []
        for row in range(self.maxRow):
            for col in range(self.maxCol):

                cur_state = self.m[0][row][col]
                cur_occ = self.m[1][row][col]
                # print("cur_state",cur_state)

                if cur_occ == self.not_occupied:

                    if cur_state == ter_state:
                        stay_state = cur_state
                        stay_occ = cur_occ
                        right_state = cur_state
                        right_occ = cur_occ
                        top_state = cur_state
                        top_occ = cur_occ
                        left_state = cur_state
                        left_occ = cur_occ
                        bottom_state = cur_state
                        bottom_occ = cur_occ

                    else:
                        temp_stay_occ = self.m[1][row][col]
                        if temp_stay_occ == self.not_occupied:
                            stay_state = cur_state
                            stay_occ = cur_occ

                        if cur_state in [15,31,47,63,79,95,111,127,143,159,175,191,207,223,239,255,]:
                            temp_stay_occ = self.m[1][row][col]
                        else:
                            # print(cur_state)
                            # print(self.m[1][row][col + 1])
                            temp_right_occ = self.m[1][row][col + 1]
                        if cur_state in [15,31,47,63,79,95,111,127,143,159,175,191,207,223,239,255,]:
                            right_state = cur_state
                            right_occ = cur_occ
                        elif temp_right_occ == self.not_occupied:
                            right_state = self.m[0][row][col + 1]
                            right_occ = self.m[1][row][col + 1]
                        else:
                            right_state = cur_state
                            right_occ = cur_occ

                        if cur_state <= 15:
                            temp_stay_occ = self.m[1][row][col]
                        else:
                            temp_top_occ = self.m[1][row - 1][col]
                        if cur_state <= 15:
                            top_state = cur_state
                            top_occ = cur_occ
                        elif temp_top_occ == self.not_occupied:
                            top_state = self.m[0][row - 1][col]
                            top_occ = self.m[1][row - 1][col]
                        else:
                            top_state = cur_state
                            top_occ = cur_occ
                            
                        if cur_state in [0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,]:
                            temp_left_occ = self.m[1][row][col]
                        else:
                            temp_left_occ = self.m[1][row][col - 1]
                        if cur_state in [0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,]:
                            left_state = cur_state
                            left_occ = cur_occ
                        elif temp_left_occ == self.not_occupied:
                            left_state = self.m[0][row][col - 1]
                            left_occ = self.m[1][row][col - 1]
                        else:
                            left_state = cur_state
                            left_occ = cur_occ

                        if cur_state >=240:
                            temp_bottom_occ = self.m[1][row][col]
                        else:
                            temp_bottom_occ = self.m[1][row + 1][col]
                        if cur_state >=240:
                            bottom_state = cur_state
                            bottom_occ = cur_occ
                        elif temp_bottom_occ == self.not_occupied:
                            bottom_state = self.m[0][row + 1][col]
                            bottom_occ = self.m[1][row + 1][col]
                        else:
                            bottom_state = cur_state
                            bottom_occ = cur_occ

                    action_map = [right_state, top_state, left_state, bottom_state, stay_state]
                    occ_map = [right_occ, top_occ, left_occ, bottom_occ, stay_occ]
                    # print(action_map)
                    for action in range(self.n_actions):

                        occ_map_rot = self.rotate(occ_map, action)
                        action_map_rot = self.rotate(action_map, action)
                        prob_sum = 0
                        for inner_action in range(self.n_actions):

                            # assign the probability of correct prob to the state in the direction of action
                            if inner_action == 0:

                                self.st[action][int(cur_state)][
                                    int(action_map_rot[inner_action])
                                ] = self.cor_pr
                                prob_sum += self.cor_pr

                            elif inner_action != 2:

                                self.st[action][int(cur_state)][
                                    int(action_map_rot[inner_action])
                                ] = self.wr_pr
                                prob_sum += self.wr_pr
        # print(self.st)
        for action in range(self.n_actions):

            rowSum = np.sum(self.st[action][:][:], axis=1)
            zeroInd = np.where(rowSum == 0)[0]
            lessInd = np.where(rowSum < 1)[0]
           

            # Assigning probability to unreachable states so that sum of each row of st matrix becomes 1
            for row in zeroInd:
                # print(row)
                col = row  # col = random.choice(range(0, self.num_states - 1))
                self.st[action][row][col] = 1
            zeroInd = np.where(rowSum == 0)[0]
            # print(zeroInd)
            # If the sum of probability for each row is not one assigning the 1 - total probability to some random state
            for row in lessInd:
                # print(row)
                col = row  # col = random.choice(range(0, self.num_states - 1))
                self.st[action][row][col] += 1 - np.sum(self.st[action][row][:])
        # print(self.st)
        return None



    def getgoal(self):
        goal_state = self.m[0][self.goalRow][self.goalCol]

        return goal_state

    def build_reward_matrix(self):

        """
        Function that builds the reward model of the world for input to the MDP for path planning

        """
        right_action = 0
        top_action = 1
        left_action = 2
        bottom_action = 3
        stay_action = 4

        goal_state = self.m[0][self.goalRow][self.goalCol]
        # print(self.maxRow)
        # print(self.maxCol)

        for row in range(self.maxRow):

            for col in range(self.maxCol):

                cur_occ = self.m[1][row][col]
                if cur_occ == self.not_occupied:

                    cur_state = int(self.m[0][row][col])

                    if cur_state == goal_state:
                        stay_state = cur_state
                        stay_occ = cur_occ
                        self.rm[cur_state][stay_action] = self.goalReward
                        right_state = cur_state
                        right_occ = cur_occ
                        self.rm[cur_state][right_action] = self.goalReward
                        top_state = cur_state
                        top_occ = cur_occ
                        self.rm[cur_state][top_action] = self.goalReward
                        left_state = cur_state
                        left_occ = cur_occ
                        self.rm[cur_state][left_action] = self.goalReward
                        bottom_state = cur_state
                        bottom_occ = cur_occ
                        self.rm[cur_state][bottom_action] = self.goalReward
                    else:
                        stay_state = cur_state
                        if cur_state in [15,31,47,63,79,95,111,127,143,159,175,191,207,223,239,255,]:
                            right_state = cur_state
                        else:
                            right_state = int(self.m[0][row][col + 1])

                        if cur_state <=15:
                            top_state = cur_state
                        else:
                            top_state = int(self.m[0][row - 1][col])

                        if cur_state in [0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,]:
                            left_state = cur_state
                        else:
                            left_state = int(self.m[0][row][col - 1])    
                        if cur_state >=240:
                            bottom_state = cur_state
                        else:
                            bottom_state = int(self.m[0][row + 1][col])     

                        stay_occ = cur_occ

                        if cur_state in [15,31,47,63,79,95,111,127,143,159,175,191,207,223,239,255,]:
                            right_occ = cur_occ
                        else:
                            right_occ = int(self.m[1][row][col + 1])

                        if cur_state <=15:
                            top_occ = cur_occ
                        else:
                            top_occ = int(self.m[1][row - 1][col])

                        if cur_state in [0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,]:
                            left_occ = cur_occ
                        else:
                            left_occ = int(self.m[1][row][col - 1])  
                        if cur_state >=240:
                            bottom_occ = cur_occ
                        else:
                            bottom_occ = int(self.m[1][row + 1][col])                  

                        
                      
                        if stay_occ != self.not_occupied:
                            self.rm[cur_state][stay_action] = self.obstReward

                        if stay_occ != goal_state:
                            self.rm[cur_state][stay_action] = self.stayReward

                        if right_occ != self.not_occupied:
                            self.rm[cur_state][right_action] = self.obstReward

                        if right_state == goal_state:
                            self.rm[cur_state][right_action] = self.moveReward

                        if top_occ != self.not_occupied:
                            self.rm[cur_state][top_action] = self.obstReward

                        if top_state == goal_state:
                            self.rm[cur_state][top_action] = self.moveReward

                        if left_occ != self.not_occupied:
                            self.rm[cur_state][left_action] = self.obstReward

                        if left_state == goal_state:
                            self.rm[cur_state][left_action] = self.moveReward

                        if bottom_occ != self.not_occupied:
                            self.rm[cur_state][bottom_action] = self.obstReward

                        if bottom_state == goal_state:
                            self.rm[cur_state][bottom_action] = self.moveReward

        # print(self.rm)
        return None

    def choose_random_nearby_state(self, cur1_state):
       
        self.build_map()
        possible_states = []
        # print(" Current state from which agent has to swerve: ", cur1_state)
        for row in range(self.maxRow):

            for col in range(self.maxCol):

                cur_state = self.m[0][row][col]
                cur_occ = self.m[1][row][col]

                # print("Cur_state: ", cur_state, "Cur1_state: ",cur1_state,"Curr_Occ: ",cur_occ)
                if cur_state == cur1_state:
                    #TODO: Stay action needed ? 

                    right_state = self.m[0][row][col + 1]
                    right_occ = self.m[1][row][col + 1]
                    if right_occ == 0:
                        # print("right state is ", right_state)
                        possible_states.append((right_state, 0))

                    top_state = self.m[0][row - 1][col]
                    top_occ = self.m[1][row - 1][col]
                    if top_occ == 0:
                        # print("top state is ", top_state)
                        possible_states.append((top_state, 1))

                    left_state = self.m[0][row][col - 1]
                    left_occ = self.m[1][row][col - 1]
                    if left_occ == 0:
                        # print("left_state is ", left_state)
                        possible_states.append((left_state, 2))

                    bottom_state = self.m[0][row + 1][col]
                    bottom_occ = self.m[1][row + 1][col]
                    if bottom_occ == 0:
                        # print("bottom_state of is ", bottom_state)
                        possible_states.append((bottom_state, 3))

        state_chosen = random.sample(possible_states, 1)
        # print("new state chopsen is: ", state_chosen[0])
        return state_chosen[0]

    def build_world(self):

        size = 16
        seed = 1337

        self.get_obstacles()
        self.build_map()
        world_data = {}
        world_data["goalRow"] = self.goalRow
        world_data["goalCol"] = self.goalCol
        terminal_states = [(world_data["goalRow"], world_data["goalCol"])]
        self.build_st_trans_matrix(terminal_states)
        self.build_reward_matrix()

        world_data["st"] = self.st.tolist()
        world_data["rm"] = self.rm.tolist()
        world_data["gamma"] = self.gamma
        world_data["num_states"] = self.num_states
        world_data["startRow"] = self.startRow
        world_data["startCol"] = self.startCol
        world_data["goalRow"] = self.goalRow
        world_data["goalCol"] = self.goalCol
        world_data["oCol"] = self.oCol
        world_data["oRow"] = self.oRow
        world_data["m"] = self.m.tolist()
        world_data["maxRow"] = self.maxRow
        world_data["maxCol"] = self.maxCol

        world_data["st"] = np.array(world_data["st"])
        world_data["rm"] = np.array(world_data["rm"])
        world_data["m"] = np.array(world_data["m"])

        terminal_states = [(world_data["goalRow"], world_data["goalCol"])]
        # print(terminal_states)
        P = world_data["st"]
        p0 = np.ones(P.shape[1]) / P.shape[1]
        return (
            world_data["st"],
            world_data["rm"],
            world_data["gamma"],
            world_data["num_states"],
            terminal_states,
            size,
            p0,
            0.0001,
            100,
            seed,
        )

    def get_initialstate(self):
        self.build_map()
        initial_state = self.m[0][self.startRow][self.startCol]
        return initial_state

    def choose_action(optimal_policy, state):
        action = optimal_policy.get(state)
        return action

    def get_policy(self):
        (
            world_data["st"],
            world_data["rm"],
            world_data["gamma"],
            world_data["num_states"],
            terminal_states,
            size,
            p0,
            epsilon,
            max_iter,
            seed,
        ) = self.build_world()
        fit_value(
            world_data["st"],
            world_data["rm"],
            world_data["gamma"],
            world_data["num_states"],
        )
        data_policy, optimal_policy, cell_values = fit_policy(
            world_data["st"],
            world_data["rm"],
            world_data["gamma"],
            world_data["num_states"],
        )

        print(
            "******************GRID WORLD******************",
            file=open("testing_output.txt", "a"),
        )
        print(self.m[1], file=open("testing_output.txt", "a"))
       
        print(
            "****************** OPTIMAL POLICY************************",
            file=open("testing_output.txt", "a"),
        )
        print(optimal_policy, file=open("testing_output.txt", "a"))
        print(
            "****************** CELL VALUES************************",
            file=open("testing_output.txt", "a"),
        )
        
        return optimal_policy, cell_values, terminal_states

    def get_rwd(self):

        (
            world_data["st"],
            world_data["rm"],
            world_data["gamma"],
            world_data["num_states"],
            terminal_states,
            size,
            p0,
            epsilon,
            max_iter,
            seed,
        ) = self.build_world()
        # print(Env(md.MDP(world_data['st'],world_data['rm'],world_data['gamma'],terminal_states,size,p0,epsilon = 0.0001, max_iter = 100,seed=seed)))
        return md.MDP(
            world_data["st"],
            world_data["rm"],
            world_data["gamma"],
            terminal_states,
            size,
            p0,
            epsilon=0.0001,
            max_iter=100,
            seed=seed,
        )

    def get_mdp(self):

        (
            world_data["st"],
            world_data["rm"],
            world_data["gamma"],
            world_data["num_states"],
            terminal_states,
            size,
            p0,
            epsilon,
            max_iter,
            seed,
        ) = self.build_world()

        # print(Env(md.MDP(world_data['st'],world_data['rm'],world_data['gamma'],terminal_states)),file=open("testing_output.txt", "a"))
        return md.MDP(
            world_data["st"],
            world_data["rm"],
            world_data["gamma"],
            terminal_states,
            size,
            p0,
            epsilon=0.0001,
            max_iter=100,
            seed=seed,
        )

    def choose_initialaction(self, agent):
        initial_state[agent] = AgentState.get_initialstate(self)

        current_state[agent] = initial_state[agent]
        Env.set_current_state_to(agent, current_state[agent])
        agent_action = AgentState.choose_action(
            initial_state[agent], policy.optimal_policy[agent]
        )
        return agent_action

    def choose(self, agent):
        agent_action = AgentState.choose_action(self, policy.optimal_policy[agent])
        return agent_action

    def choose_action(self, optimal_policy):
        action = optimal_policy.get(self)
        return action

    
    def initial_setup():
        # env_params = set_env_params()
        # agents_param=set_agents()
        term = {}
        for agent in agents:
            agent_gridworld[agent] = define_gridworld(agent)
            agent_mdp[agent] = AgentState.get_mdp(agent_gridworld[agent])
        
            agent_mdp_individual_reward[agent] = AgentState.get_mdp(
                agent_gridworld[agent]
            )
            (
                optimal_policy[agent],
                cell_values[agent],
                term[agent],
            ) = AgentState.get_policy(agent_gridworld[agent])
            
            goal[agent] = AgentState.getgoal(agent_gridworld[agent])

     
        policy = PolicyAgent(optimal_policy)
        env = Env(agent_mdp)
        # print(type(env))
        return agent_gridworld, policy, env

class Env(AgentState, MDP):
    def __init__(self, agent_mdp):
        self.agent_mdp = agent_mdp
        self.agent_mdp_individual_reward = agent_mdp_individual_reward

    def remove_agent(agent):
        agent_mdp.pop(agent)
        agents.remove(agent)
        if agent in agent_action:
            agent_action.pop(agent)
        if agent in current_state:
            current_state.pop(agent)
        if agent in agent_section:
            agent_section.pop(agent)
        if agent in agents_param:
            agents_param.pop(agent)
        if agent in temp_agent_state:
            temp_agent_state.pop(agent)

    
    def check_agents_nearby(current_agent_1,current_agent_2):
        coordinates_nearto_givenstate = []
        nearby_states_list = []
        other_agents_loc = []
        diagonal_nearby_coords = []
        agents_current_pos = {}
        
        for agent in agents:
            agent_state = agent_mdp.get(agent)
            if agent_state is None:
                # print(f"agent_mdp does not contain key: {agent}")
                continue
            # print("agent_mdp[agent]")
            # print(agent_state)
            agents_current_pos[agent] = agent_state.get_current_state_idx()
        
        # print("agents_current_pos in check_agent_nearbyfunc", agents_current_pos)

        if len(agents_current_pos) == 2:
            NE_cur_state = agents_current_pos.get(current_agent_1)
            if NE_cur_state is None:
                # print(f"NE_agent ({NE_agent}) not in agents_current_pos")
                return None, None, [], [], None, None, None, None

            (
                coordinates_nearto_givenstate,
                colstartidx,
                colendidx,
                rowstartidx,
                rowendidx,
                diagonal_nearby_coords,
            ) = Env.get_adjacent_cells(agent_gridworld[current_agent_1], NE_cur_state)

            other_agents_cur_state = agents_current_pos.get(current_agent_2)
            if other_agents_cur_state is None:
                # print(f"other_agent ({other_agent}) not in agents_current_pos")
                return None, None, [], [], None, None, None, None

            other_agents_coordinate = AgentState.get_coordinates_ofgiven_state(
                agent_gridworld[current_agent_2], other_agents_cur_state
            )
            
            if (
                other_agents_coordinate[0][0] in [coord[0] for coord in diagonal_nearby_coords]
                and other_agents_coordinate[0][1] in [coord[1] for coord in diagonal_nearby_coords]
            ) or (
                other_agents_coordinate[0][0] in [coord[0] for coord in coordinates_nearto_givenstate]
                and other_agents_coordinate[0][1] in [coord[1] for coord in coordinates_nearto_givenstate]
            ):
                
                other_agents_loc.append(other_agents_cur_state)
                        
            return (
                current_agent_1,
                current_agent_2,
                NE_cur_state,
                other_agents_loc,
                colstartidx,
                colendidx,
                rowstartidx,
                rowendidx,
            )
        else:
            return None, None, [], [], None, None, None, None


    def get_adjacent_cells(self, cur1_state):
        coordinates_ofgiven_state = self.get_coordinates_ofgiven_state(cur1_state)
        # coordinates_ofgiven_state = self.get_coordinates_ofgiven_state(34)
        (
            coordinates_nearto_givenstate,
            colstartidx,
            colendidx,
            rowstartidx,
            rowendidx,
            diagonal_nearby_coords,
        ) = self.get_coordinates_nearto_givenstate(coordinates_ofgiven_state)
        return (
            coordinates_nearto_givenstate,
            colstartidx,
            colendidx,
            rowstartidx,
            rowendidx,
            diagonal_nearby_coords,
        )

    def set_current_state_to(agent, state):
        agent_mdp[agent].set_current_state_to(state)

    def set_agent_step_to(self, agent, old_agent_step, new_agent_state, action):
        # print("old state: ", old_agent_step)
        # print("new state: ", new_agent_state)
        new_agent_step = agent_mdp[agent].set_agent_step_to(
            old_agent_step, new_agent_state, action
        )
        return new_agent_step

    def single_agent_set_current_state_to(agent, state):
        agent_mdp_individual_reward[agent].set_current_state_to(state)

    def reset(self):
        for agent in agents_count:
            agent_gridworld[agent] = define_gridworld(agent)
            agent_mdp[agent] = AgentState.get_mdp(agent_gridworld[agent])
        env = Env(agent_mdp)
        return agent_gridworld

    def check_collision(self, current_state):
        rev_dict = {}
        result = {}
        lstresult = []
        collidingpair = []
        # print("CURRENT_state in env step")
        # print(current_state)
        # print("self in env step")
        # print(self)
        for key, val in self.items():
            for oldkey, oldval in current_state.items():
                if val[0] == oldval:
                    if key == oldkey:
                        pass
                    else:
                        collidingpair.append((key, oldkey))

        if len(collidingpair) > 0:
           
            lst = []
            for i in collidingpair:
                for j in collidingpair:
                    if i[0] == j[1] and i[1] == j[0]:
                        collidingpair.remove(i)
                        lst.append(i)
            lstresult = lst

    
        for key, value in self.items():
            rev_dict.setdefault(value[0], set()).add(key)

            result = set(
                chain.from_iterable(
                    values for key, values in rev_dict.items() if len(values) > 1
                )
            )
     
        return (lstresult, list(result))

  
    def single_agent_step(self, action):
        single_agent_step = self.agent_mdp_individual_reward[agent].step(action)
        return single_agent_step

    def step(self, n_action):
        potential_step_from_common = {}
        first_possible_step_copy = {}
        random_agent = []

        for agent in self.agent_mdp:
            potential_step_from_common[agent] = self.agent_mdp[agent].step(
                n_action[agent]
            )
        lstresult, result = Env.check_collision(
            potential_step_from_common, current_state
        )
        if lstresult == [] and result == []:
            return potential_step_from_common
        else:
            new_state[agent] = current_state[
                agent
            ]  # since both agents trying to move to same state, bouncing back
            first_possible_step_copy[agent] = env.set_agent_step_to(
                agent,
                current_state[agent],
                new_state[agent],
                n_action[agent],
            )
            Env.set_current_state_to(agent, first_possible_step_copy[agent][0])
            # print("first_possible_step_copy: ", first_possible_step_copy)
            possible_new_agent_step = first_possible_step_copy


            return possible_new_agent_step
 
class PolicyAgent(AgentState):
    def __init__(self, optimal_policy):
        self.optimal_policy = optimal_policy


def choose_startrow(agent, pattern):
    if pattern == "parallel":
        if agent == 0:
            return 1
        if agent == 1:
            return 1
    elif pattern == "extended_parallel":
        if agent == 0:
            return 1
        if agent == 1:
            return 1
def choose_startcol(agent, pattern):
    if pattern == "parallel":
        if agent == 0:
            return 1
        if agent == 1:
            return 4
    elif pattern == "extended_parallel":
        if agent == 0:
            return 6
        if agent == 1:
            return 10
def choose_goalrow(agent, pattern):
    if pattern == "parallel":
        if agent == 0:
            return 4
        if agent == 1:
            return 4
    elif pattern == "extended_parallel":
        if agent == 0:
            return 4
        if agent == 1:
            return 4
def choose_goalcol(agent, pattern):
    if pattern == "parallel":
        if agent == 0:
            return 4
        if agent == 1:
            return 1
    elif pattern == "extended_parallel":
        if agent == 0:
            return 10
        if agent == 1:
            return 6

def get_agent_state():
    for agent, value in agents_param.items():
        temp_agent_state[agent] = (value.get("startRow"), value.get("startCol"))
    return None

def check_celloccupancy():
    for current_agent, current_agent_val in temp_agent_state.items():
        for next_agent, next_agent_val in temp_agent_state.items():
            if temp_agent_state[current_agent] == temp_agent_state[next_agent]:
                if current_agent == next_agent:
                    pass
                else:
                    return next_agent

def define_gridworld(agent):
    world_params = env_params.copy()
    world_params.update(agents_param[agent])
    return AgentState(**world_params)

def is_goal(agent_mdp):
    mdp_done_values = {}
    finished = {}
    agents_copy = agents.copy()
    for agent in agents_copy:
        mdp_done_values[agent] = agent_mdp[agent].done
        if agent_mdp[agent].done == True:
            finished[agent] = True
            Env.remove_agent(agent)

    if all(value == True for value in mdp_done_values.values()):
        return True
    else:
            return False

def set_env_params():
    env_params = {
        "maxRow": 16,
        "maxCol": 16,
        "num_obstacle_pts": 3,
        "cor_pr": 1.0,
        "wr_pr": 0.0,
        "n_actions": 5,
        "goalReward": 0,
        "obstReward": -1,
        "moveReward": -1,
        "stayReward": -0.5,
        "gamma": 1
    }
    return env_params

def extract_agent_plans(plan):
                dict_aplan = {}
                dict_bplan = {}
                current_phase = 0

                for phase in plan:
                    if phase is None:
                        dict_aplan[current_phase] = []
                        dict_bplan[current_phase] = []
                        current_phase += 1
                    else:
                        for agent, action in phase:
                            if action in action_mapping:
                                action_code = action_mapping[action]
                                if agent == "A":
                                    dict_aplan[current_phase - 1].append(action_code)
                                elif agent == "B":
                                    dict_bplan[current_phase - 1].append(action_code)

                return dict_aplan, dict_bplan

def phase2(pattern,current_agent_1,current_agent_2, all_agents, norm_emerged_csms):

    print("pattern,current_agent_1,current_agent_2, all_agents, norm_emerged_csms", pattern,current_agent_1,current_agent_2, all_agents, norm_emerged_csms)
    for agent in agents:
        agent_action[agent] = AgentState.choose_initialaction(agent_gridworld[agent], agent)
    agent_initial_action = agent_action

    for agent in agents:
        value[agent] = 0
        gamma_val[agent] = 0
        cost[agent] = tuple()

    agent_gamma = {}
    agent_gamma_dict = {}

    while is_goal(env.agent_mdp) == False:
        other_agents_state = []
        new_plan = []
        print("current_agent_1, current_agent_2", current_agent_1, current_agent_2)
        agents_for_ROC_copy = [current_agent_1, current_agent_2].copy()       
        (
            NE_agent,
            other_agent,
            NE_cur_state,
            other_agents_state,
            colstartidx,
            colendidx,
            rowstartidx,
            rowendidx,
        ) = Env.check_agents_nearby(current_agent_1, current_agent_2)
        (current_agent_1, current_agent_2, current_agent_1_cur_state, other_agents_state, colstartidx, colendidx, rowstartidx, rowendidx) = Env.check_agents_nearby(current_agent_1, current_agent_2)
        print(current_agent_1, current_agent_2, current_agent_1_cur_state, other_agents_state, colstartidx, colendidx, rowstartidx, rowendidx)
        
        try:
            if not other_agents_state:
                print("No other agent")
        except ValueError as e:
            print(f"Skipping to the next iteration: {str(e)}")
            return False
        
        # agents_for_ROC_copy = agents.copy()       
        # (current_agent_1,other_agent,NE_cur_state,other_agents_state,colstartidx,colendidx,rowstartidx,rowendidx,) = Env.check_agents_nearby()
        if other_agents_state:
                state_matrix, ROC, NE_cur_state, colstartidx, colendidx, rowstartidx, rowendidx = AgentState.get_ROC_matrix(agent_gridworld[current_agent_1],NE_cur_state,colstartidx,colendidx,rowstartidx,rowendidx,)
                grid_world = AgentState.get_world(agent_gridworld[current_agent_1])
                ROC_grid_world = AgentState.get_grid_world_ROC(grid_world[1], colstartidx, colendidx, rowstartidx, rowendidx)
                print(ROC_grid_world, file=open("testing_output.txt", "a"))
                (NE_agent_state_cell_value_matrix,NE_agent_ROC_cell_value_matrix,) = AgentState.get_ROC_cell_value_matrix(agent_gridworld[current_agent_1],current_agent_1,NE_cur_state,colstartidx,colendidx, rowstartidx, rowendidx,)
                ( other_agent_state_cell_value_matrix,other_agent_ROC_cell_value_matrix,) = AgentState.get_ROC_cell_value_matrix(agent_gridworld[current_agent_1],other_agent,NE_cur_state,colstartidx,colendidx,rowstartidx,rowendidx,)
                combined_array = AgentState.get_combinedagent_matrix(state_matrix,ROC,grid_world,ROC_grid_world,NE_cur_state,other_agents_state,)
                combined_array = combined_array.tolist()
                NE_agent_ROC_cell_value_matrix = NE_agent_ROC_cell_value_matrix.tolist()
                other_agent_ROC_cell_value_matrix = (other_agent_ROC_cell_value_matrix.tolist())
                grid = np.array(combined_array)
                NE_agent_ROC_cell_value_matrix = np.array(NE_agent_ROC_cell_value_matrix)
                other_agent_ROC_cell_value_matrix = np.array(other_agent_ROC_cell_value_matrix)
                NE_Coord = AgentState.get_coordinates_ofgiven_state(agent_gridworld[current_agent_2], NE_cur_state)
                agA_startrow,agA_startcol = NE_Coord[0]
                other_agent_Coord = AgentState.get_coordinates_ofgiven_state(agent_gridworld[current_agent_2], other_agents_state)
                agB_startrow,agB_startcol = other_agent_Coord[0]
                
                print(NE_agent_ROC_cell_value_matrix)
                print(other_agent_ROC_cell_value_matrix)
                print("NE_Coord",NE_Coord)
                print("other_agent_Coord",other_agent_Coord)
                NEROC_cellvalue_matrix = np.abs(NE_agent_ROC_cell_value_matrix)
                otheragentROC_cellvalue_matrix = np.abs(other_agent_ROC_cell_value_matrix)
                

                #FIND REGLANDSCAPE AND THEN THRESHOLD LANDSCAPE TO CHECK IF AG HAS A SOLUTION CSM ALREADY WITH IT
                regret_landscapeA, regret_landscapeB = reg_calc(agA_startrow, agA_startcol, NE_cur_state, NEROC_cellvalue_matrix, agB_startrow, agB_startcol,other_agents_state, otheragentROC_cellvalue_matrix)
                threshold_regA,threshold_regB = threshold_reg(regret_landscapeA,regret_landscapeB)

                for agent_id in agents:
                    agent_objects[agent_id] = Agent(agent_id)

                agent_objects['current_agent_1'] = agent_objects[current_agent_1]
                agent_objects['current_agent_2'] = agent_objects[current_agent_2]
                agent_objects['current_agent_1'] = all_agents[current_agent_1]
                agent_objects['current_agent_2'] = all_agents[current_agent_2]

                contourtree_G, contour_tree = gen_contour(threshold_regA, threshold_regB)
                new_tree_id = f"tree_{agent_id}_{random.randint(1000, 9999)}"
               
                print(f"Contour tree saved with ID: {new_tree_id}")

                
                print(current_agent_1)
                all_trees = agent_objects['current_agent_1'].get_all_contour_trees()
                print(len(all_trees))
                print(all_trees)

                # Compare the new tree with existing trees
                similar_trees = []
                if len(all_trees) > 0:  # Check if there are any other trees besides the new one
                    for tree_id, existing_tree in all_trees.items():
                        print(new_tree_id, tree_id)
                        if tree_id != new_tree_id:  # Don't compare with itself
                            if compare_contour_trees(contourtree_G, existing_tree):
                                similar_trees.append(tree_id)

                    # Print similar trees if any
                    if similar_trees:
                        print(f"Similar trees found: {', '.join(similar_trees)}")
                        print(f"Similar trees found: {', '.join(similar_trees)}",file=open("norm_emergence_output.txt", "a"))
                    else:
                        print("No similar trees found")
                        print("No similar trees found",file=open("norm_emergence_output.txt", "a"))
                        print("This is the first tree in the agent's memory")
                        print("No similar trees found",file=open("norm_emergence_output.txt", "a"))

                initial = agentgrid.AgentGridState(                 grid,                      NE_agent_ROC_cell_value_matrix,                   other_agent_ROC_cell_value_matrix,                   )
                goal = agentgrid.AgentGridState.definegoal(                        initial,                        grid,                        NE_agent_ROC_cell_value_matrix,                        other_agent_ROC_cell_value_matrix,                    )
                agentgrid.AgentGridState(initial,                        NE_agent_ROC_cell_value_matrix,                        other_agent_ROC_cell_value_matrix,                    )
                        
                print("$$$$$$$$$ DETAILS $$$$$$$")
                print("initial", initial)
                print("AGENTS_PARAM", agents_param)
                
                resultA = np.where(np.vectorize(lambda x: 'A' in x)(initial.grid))
                row_A, col_A = resultA[0][0], resultA[1][0]
                start_coord_A = (row_A, col_A)
                resultB = np.where(np.vectorize(lambda x: 'B' in x)(initial.grid))
                row_B, col_B = resultB[0][0], resultB[1][0]
                start_coord_B = (row_B, col_B)
                grow_A= (agent_gridworld[current_agent_1].goalRow)
                gcol_A= (agent_gridworld[current_agent_1].goalCol)-1
                grow_B= (agent_gridworld[current_agent_2].goalRow)
                gcol_B= (agent_gridworld[current_agent_2].goalCol)-1
                end_coord_A = (grow_A,gcol_A)
                end_coord_B = (grow_B,gcol_B)
                print("ENDCOORDA, B", end_coord_A, end_coord_B)

                print("NE_Coord",NE_Coord)
                print("other_agent_Coord",other_agent_Coord)

                print("start_coord_A",start_coord_A)
                print("start_coord_B",start_coord_B)

                print("ENDCOORDA, B", end_coord_A, end_coord_B,file=open("norm_emergence_output.txt", "a"))

                print("NE_Coord",NE_Coord,file=open("norm_emergence_output.txt", "a"))
                print("other_agent_Coord",other_agent_Coord,file=open("norm_emergence_output.txt", "a"))

                print("start_coord_A",start_coord_A,file=open("norm_emergence_output.txt", "a"))
                print("start_coord_B",start_coord_B,file=open("norm_emergence_output.txt", "a"))
                
                
                cellvalue_A = np.abs(NE_agent_ROC_cell_value_matrix)
                cellvalue_B = np.abs(other_agent_ROC_cell_value_matrix)
                tree_id = None
                if not similar_trees:    
                        plan=[]
                        ag = agentgrid.Agentsearch(initial, goal, pattern)
                        plan = agentgrid.searchsolution(ag)
                        print("\nPLANNNNN\n",plan)
                        final_plan={}
                        count=0
                        for i in range(len(plan)):
                            temp=[]
                            if (plan[i] is None):
                                i+=1
                                while(i<len(plan)):
                                    if(plan[i] is not None):
                                        temp.append(plan[i])
                                        i+=1
                                    else:
                                        break
                                final_plan[count]=temp
                                count+=1
                        # print("\nFINAL PLAN\n",final_plan)
                        temp={}
                        sam={}
                        for key,val in final_plan.items():
                            aval=[]
                            bval=[]
                            # print("\n",key,val)
                            for tup in val:
                                # print("\n",tup)
                                for x in tup:
                                    # print("\n",x,x[0])
                                    if x[0] == 'A':
                                        if x[1]=='left' or x[1]=='right'or x[1]=='up'or x[1]=='down':
                                            cos = 1
                                        elif x[1]=='agoffgrid':
                                            cos = None
                                        elif x[1]=='stay':
                                            cos = 0.5
                                        # print(x[0],cos)
                                        if cos is not None:
                                            aval.append(cos)
                                    elif x[0] == 'B':
                                        if x[1]=='left' or x[1]=='right'or x[1]=='up'or x[1]=='down':
                                            cos = 1
                                        elif x[1]=='agoffgrid':
                                            cos = None
                                        elif x[1]=='stay':
                                            cos = 0.5
                                        # print(x[0],cos)
                                        if cos is not None:
                                            bval.append(cos)
                                        # print(aval)
                                        temp['A'] = aval
                                        temp['B'] = bval
                            # print("\nTemp",temp)
                            # print("\n sam before",sam)
                            sam[key]=temp
                            temp={}

                        # Extract agent plans from the given plan
                        dict_aplan, dict_bplan = extract_agent_plans(plan)
                        LCP_plans = {key: {'A': value_a, 'B': value_b} for key, value_a, value_b in zip(dict_aplan.keys(), dict_aplan.values(), dict_bplan.values())}
                        
                        Constrained_state_machines = {
                            new_tree_id: roc_to_csm(
                                cellvalue_A,
                                plan['A'],
                                start_coord_A,
                                end_coord_A,
                                cellvalue_B,
                                plan['B'],
                                start_coord_B,
                                end_coord_B
                            )
                            for key, plan in LCP_plans.items()
                        }
                        print(Constrained_state_machines)
                        all_agents[current_agent_1].add_contour_tree(contourtree_G, new_tree_id)
                        all_agents[current_agent_1].add_constrained_state_machines(new_tree_id, Constrained_state_machines)
                        all_agents[current_agent_2].add_contour_tree(contourtree_G, new_tree_id)
                        all_agents[current_agent_2].add_constrained_state_machines(new_tree_id, Constrained_state_machines)
                        print(f"Constrained State Machines added for tree {new_tree_id}")
                        print(f"Constrained State Machines added for tree {new_tree_id}",file=open("norm_emergence_output.txt", "a"))
                    

                        coordinator = Coordinator()
                        coordinator.run(Constrained_state_machines)
                        selected_csm_id = coordinator.get_selected_csm_id()

                        if selected_csm_id in all_agents[current_agent_1].constrained_state_machines[new_tree_id]:
                            smA, smB = all_agents[current_agent_1].constrained_state_machines[new_tree_id][selected_csm_id]
                            smA.reset()
                            smB.reset()
                            print("smA:", smA.state)
                            print("smB:", smB.state)
                        else:
                            print("Selected CSM ID not found in the dictionary.")
                
                else:

                    similar_trees_dict = {i: tree_id for i, tree_id in enumerate(similar_trees)}
                    coordinator = Coordinator()
                    coordinator.run(similar_trees_dict)
                    selected_index = coordinator.get_selected_csm_id()
                    if selected_index is not None and selected_index < len(similar_trees):
                        selected_tree_id = similar_trees[selected_index]

                        # Check if the selected tree has a CSMselected_index
                        if selected_tree_id in all_agents[current_agent_1].constrained_state_machines:
                            all_agents[current_agent_1].increment_csm_usage(selected_tree_id)
                            usage_count = all_agents[current_agent_1].get_csm_usage_count(selected_tree_id)
                            print(f"Reused CSM for tree {selected_tree_id}. Current usage count: {usage_count}")
                            print(f"Reused CSM for tree {selected_tree_id}. Current usage count: {usage_count}",file=open("norm_emergence_output.txt", "a"))
                            
                            if usage_count >= NORM_EMERGENCE_THRESHOLD:
                                print(f"Norm has emerged for CSM associated with tree {selected_tree_id}!")
                                print(f"Norm has emerged for CSM associated with tree {selected_tree_id}!",file=open("norm_emergence_output.txt", "a"))
                                norm_emerged_csms.add(selected_tree_id)

                        
                        constrained_state_machines = all_agents[current_agent_1].get_constrained_state_machines(selected_tree_id)
                        print(selected_tree_id)
                        # print(selected_csm_id)
                        print(selected_index)
                        print("constrained_state_machines",constrained_state_machines)
                        if selected_tree_id in constrained_state_machines:
                                smA, smB = constrained_state_machines[selected_tree_id]
                                smA.reset()
                                smB.reset()
                                print("smA:", smA)
                                print(smA.state)
                                print("smB:", smB)
                                print(smB.state)
                        else:
                                print("Selected CSM ID not found in the constrained state machines of the agent.")

                    else:
                        print("Invalid selection from coordinator.")


                # Assign aplan_new and bplan_new with actions of the chosen phase
                aplan = csm_to_roc(smA, start_coord_A, end_coord_A, cellvalue_A,start_coord_B,end_coord_B, cellvalue_B, 'A')
                bplan = csm_to_roc(smB, start_coord_A, end_coord_A, cellvalue_A,start_coord_B,end_coord_B, cellvalue_B, 'B')

                print("aplan:", aplan)
                print("bplan:", bplan)

                new_aplan = aplan.copy()
                new_bplan = bplan.copy()
                while new_aplan or new_bplan:
                    agent_map = {0: current_agent_1, 1: current_agent_2}
                    
                    if current_agent_1 not in agents:
                        new_aplan = []
                    if current_agent_2 not in agents:
                        new_bplan = []

                    if new_aplan:
                        if current_agent_1 in agents:
                            agent_planaction[current_agent_1] = new_aplan[0]
                            new_aplan.remove(new_aplan[0])
                    else:
                        if current_agent_1 in agents:
                            agent_planaction[current_agent_1] = AgentState.choose(
                                agent_step[current_agent_1][0], current_agent_1
                            )

                    if new_bplan:
                        if current_agent_2 in agents:
                            agent_planaction[current_agent_2] = new_bplan[0]
                            new_bplan.remove(new_bplan[0])
                    else:
                        if current_agent_2 in agents:
                            agent_planaction[current_agent_2] = AgentState.choose(
                                agent_step[current_agent_2][0], current_agent_2
                            )

                    # print(agent_planaction)
                    if iter_id not in simulate_actions:
                        simulate_actions[iter_id] = {}
                    for agent, action in agent_planaction.items():
                        if agent in simulate_actions[iter_id]:
                            simulate_actions[iter_id][agent].append(action)
                        else:
                            simulate_actions[iter_id][agent] = [action]
                    # print(simulate_actions) 

                    agent_step = env.step(agent_planaction)

                    agent_planaction.clear()
                    for agent in agent_step:
                        current_agent_step_tracking[agent] = agent_step[agent]

                    for agent in agent_step:
                        Env.set_current_state_to(agent, agent_step[agent][0])
                    for agent in agent_step:
                        agent_gamma[agent] = agent_step[agent][3]
                    for agent, val in agent_gamma.items():
                        for key in val:
                            agent_gamma_dict[agent] = val[key]
                    for agent in agent_step:
                        current_state[agent] = agent_step[agent][0]
                        value[agent] = value[agent] + agent_step[agent][1]

                        gamma_val[agent] = gamma_val[agent] + agent_gamma_dict[agent]
                        accumulated_reward[agent] = value[agent] - gamma_val[agent]
                    is_goal(env.agent_mdp)
                    if not new_aplan and not new_bplan:
                        agent_action.clear()

                else:
                    if agent_action:
                        # print(agent_action)
                        if iter_id not in simulate_actions:
                            simulate_actions[iter_id] = {}
                        for agent, action in agent_action.items():
                            if agent in simulate_actions[iter_id]:
                                simulate_actions[iter_id][agent].append(action)
                            else:
                                simulate_actions[iter_id][agent] = [action]
                        # print(simulate_actions)                        
                        agent_step = env.step(agent_action)
                    else:
                        for agent in agents:
                            agent_action[agent] = AgentState.choose(agent_step[agent][0], agent)
                        if iter_id not in simulate_actions:
                            simulate_actions[iter_id] = {}
                        for agent, action in agent_action.items():
                            if agent in simulate_actions[iter_id]:
                                simulate_actions[iter_id][agent].append(action)
                            else:
                                simulate_actions[iter_id][agent] = [action]
                        # print(simulate_actions) 
                        agent_step = env.step(agent_action)
                        

                    for agent in agent_step:
                        current_agent_step_tracking[agent] = agent_step[agent]

                    for agent in agent_step:
                        Env.set_current_state_to(agent, agent_step[agent][0])
                    for agent in agent_step:
                        agent_gamma[agent] = agent_step[agent][3]
                    for agent, val in agent_gamma.items():
                        for key in val:
                            agent_gamma_dict[agent] = val[key]
                    for agent in agent_step:
                        current_state[agent] = agent_step[agent][0]
                        value[agent] = value[agent] + agent_step[agent][1]

                        gamma_val[agent] = gamma_val[agent] + agent_gamma_dict[agent]
                        accumulated_reward[agent] = value[agent] - gamma_val[agent]
                    for agent in agents:
                        agent_action[agent] = AgentState.choose(agent_step[agent][0], agent)

                print("***********************************END OF PHASE2********************************",file=open("norm_emergence_output.txt", "a"))
                print("***********************************END OF PHASE2********************************")
                print("***********************************END OF PHASE2********************************")
        return True

def clean_env():
    world_data.clear()
    finished = False
    agents.clear()
    agents_individual_reward.clear()
    agents_count.clear()
    world_params.clear()
    optimal_policy.clear()
    agent_mdp.clear()
    agent_mdp_individual_reward.clear()
    agent_step.clear()
    initial_state.clear()
    agent_action.clear()
    agent_action_individual_reward.clear()
    current_state.clear()
    new_state.clear()
    goal.clear()
    agent_gridworld.clear()
    agent_gridworld_individual_reward.clear()
    accumulated_reward.clear()
    accumulated_reward_individual.clear()
    agent_reward.clear()
    value.clear()
    gamma_val.clear()
    agent_gamma.clear()
    agent_gamma_dict.clear()
    agent_gamma_individual_reward.clear()
    agent_gamma_individual_reward_dict.clear()
    agents_param.clear()
    agent_section.clear()
    temp_agent_state.clear()

def average_calc():
    res = dict()
    temp = dict()
    for sub in difference_of_reward.values():
        for key, ele in sub.items():
            res[key] = ele + res.get(key, 0)
            temp[key] = res[key] / len(difference_of_reward)
    return temp

TOTAL_AGENTS = 25
AGENTS_PER_ITERATION = 2
NORM_EMERGENCE_THRESHOLD = 8
max_iterations = 100
difference_of_reward = {}
patterns = ["parallel", "extended_parallel"]

def set_agents_param(iteration, current_agent_1, current_agent_2):
    print("AT SET_AGENTS PARAM")
    pattern = random.choice(patterns)  # Choose randomly from the defined patterns
    agents.add(current_agent_1)
    agents.add(current_agent_2)
    agents_individual_reward.add(current_agent_1)
    agents_individual_reward.add(current_agent_2)
    agents_count.append(current_agent_1)
    agents_count.append(current_agent_2)
    agents_param.update(
        {
            current_agent_1: {
                "startRow": choose_startrow(0, pattern),
                "startCol": choose_startcol(0, pattern),
                "goalRow": choose_goalrow(0, pattern),
                "goalCol": choose_goalcol(0, pattern),
            },
            current_agent_2: {
                "startRow": choose_startrow(1, pattern),
                "startCol": choose_startcol(1, pattern),
                "goalRow": choose_goalrow(1, pattern),
                "goalCol": choose_goalcol(1, pattern),
            }
        }
    )

    get_agent_state()
    return pattern

def create_network_graph(usage_data, norm_threshold):
    G = nx.Graph()
    for tree_id, count in usage_data.items():
        G.add_node(tree_id, size=count, color='green' if count >= norm_threshold else 'gray')
    
    # Add edges between all nodes (you can modify this based on your needs)
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                G.add_edge(node1, node2, weight=min(usage_data[node1], usage_data[node2]))

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    
    node_sizes = [G.nodes[node]['size'] * 100 for node in G.nodes()]
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight'] * 0.1 for u, v in G.edges()])
    nx.draw_networkx_labels(G, pos)
    
    plt.title("Norm Emergence Network Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("norm_emergence_network.png")
    plt.close()

def create_animated_bubble_chart(usage_data_over_time, norm_threshold):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def update(frame):
        ax.clear()
        data = usage_data_over_time[frame]
        x = list(range(len(data)))
        y = list(data.values())
        sizes = [count * 20 for count in y]
        colors = ['green' if count >= norm_threshold else 'gray' for count in y]
        
        ax.scatter(x, y, s=sizes, c=colors, alpha=0.6)
        
        for i, (tree_id, count) in enumerate(data.items()):
            ax.annotate(tree_id, (i, count), ha='center', va='center')
        
        ax.set_xlabel('Tree ID')
        ax.set_ylabel('Usage Count')
        ax.set_title(f'Norm Emergence - Iteration {frame + 1}')
        ax.axhline(y=norm_threshold, color='r', linestyle='--', label='Norm Threshold')
        ax.legend()
        
    anim = FuncAnimation(fig, update, frames=len(usage_data_over_time), interval=500, repeat=False)
    anim.save('norm_emergence_animation.gif', writer='pillow')
    plt.close()

def create_line_graph(usage_data_over_time, norm_threshold):
    plt.figure(figsize=(12, 8))
    
    # Create a dictionary to store the usage data for each tree
    tree_usage = {}
    
    # Collect data for each tree across all iterations
    for iteration, data in enumerate(usage_data_over_time):
        for tree_id, count in data.items():
            if tree_id not in tree_usage:
                tree_usage[tree_id] = [0] * len(usage_data_over_time)
            tree_usage[tree_id][iteration] = count
    
    # Plot a line for each tree
    for tree_id, usage in tree_usage.items():
        plt.plot(range(len(usage_data_over_time)), usage, label=f'Tree {tree_id}', marker='o')
    
    # Add the norm emergence threshold line
    plt.axhline(y=norm_threshold, color='r', linestyle='--', label='Norm Threshold')
    
    plt.xlabel('Iteration')
    plt.ylabel('Usage Count')
    plt.title('Tree Usage Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('tree_usage_over_time.png')
    plt.close()

usage_data_over_time = []

# The main loop
all_agents = {i: Agent(i) for i in range(TOTAL_AGENTS)}
norm_emerged_csms = set()
# The main loop
for iter_id in range(0, max_iterations):
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& START OF ITERATION", iter_id,file=open("norm_emergence_output.txt", "a"))
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& START OF ITERATION", iter_id)
    clean_env()
    env_params = set_env_params()

    # Select agents for the current iteration
    available_agents = set(range(TOTAL_AGENTS)) - set(agents_count)
    if len(available_agents) < AGENTS_PER_ITERATION:
        available_agents = set(range(TOTAL_AGENTS))

    selected_agents = random.sample(list(available_agents), AGENTS_PER_ITERATION)
    current_agent_1, current_agent_2 = selected_agents

    pattern = set_agents_param(iter_id, current_agent_1, current_agent_2)
    print("PATTERN:", pattern)
    (
        agent_gridworld,
        policy,
        env,
    ) = AgentState.initial_setup()
    
    for agent in agents:
        print("agent gridworld of", agent)
        print(agent_gridworld[agent].startRow)
        print(agent_gridworld[agent].startCol)
        print(agent_gridworld[agent].goalRow)
        print(agent_gridworld[agent].goalRow)
    for agent in agents:
        agent_action[agent] = AgentState.choose_initialaction(
            agent_gridworld[agent], agent
        )
        agent_action_individual_reward[agent] = agent_action[agent]

    phase2_success= phase2(pattern, current_agent_1, current_agent_2, all_agents, norm_emerged_csms)

    if not phase2_success:
        print("Phase2 did not complete successfully. Moving to the next iteration.")
        continue

    current_usage_data = {
    tree_id: {
        'usage_count': all_agents[current_agent_1].get_csm_usage_count(tree_id),
        'from': current_agent_1,
        'to': current_agent_2
    }
    for tree_id in all_agents[current_agent_1].contour_trees.keys()
}

    usage_data_over_time.append(current_usage_data)

    print("current_usage_data",current_usage_data)
    print("current_usage_data",current_usage_data,file=open("norm_emergence_output.txt", "a"))
    # create_network_graph(current_usage_data, NORM_EMERGENCE_THRESHOLD)

print("usage_data_over_time",usage_data_over_time)
print("usage_data_over_time",usage_data_over_time,file=open("norm_emergence_output.txt", "a"))

