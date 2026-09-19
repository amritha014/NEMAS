
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import mdptoolbox as mdptoolbox
import collections
import csv
import time
import pickle
import json
import mdp as md
import mdp_gym.env as en
import mdp_gym.utils_gym as ut
right_action = 0
top_action = 1
left_action = 2
bottom_action = 3
world_data= {}
agents={'agent_A':1, 'agent_B':2}
world_params = {}
optimal_policy={}
agent_mdp={}
initial_env={}
agent_step={}
initial_state={}
agent_action={}

class PathPlanning:
	"""
	Class that builds the model of the world for input to the MDP for path planning
	
	Inputs:
	-------
	maxRow : (int) maximum number of columns in the map
	maxCol : (int) maximum number of columns in the map
	n_obstacle_pts : (int) number of obstacle points
	cor_pr : (float) probability to be assigned to the state transition coresponding to the correct action direction
	wr_pr : (float) probability to be assigned to the state transition corresponding to the all other action directions
			sum of cor_pr and wr_pr for all the actions should be greater equal to 0.99
	n_actions : (int) number of dimensions (directions are assumed to be in anti clockwise direction)
	startRow : (int) row of the starting state
	startCol : (int) column of the starting state
	goalRow : (int) row of the goal state
	goalCol : (int) column of the goal state
	goalReward : (int) reward when next state is goal state
	obstReward : (int) reward when next state is the obstacle 
	stayReward : (int) reward when next state neither goal state nor the obstacle 

	"""

	def __init__(self, maxRow = 50, maxCol = 50, num_obstacle_pts = 50, cor_pr = 0.7, wr_pr = 0.1, n_actions = 4, agent_A_startRow = 5, agent_A_startCol = 5, agent_A_goalRow = 45, agent_A_goalCol = 45, agent_B_startRow = 5, agent_B_startCol = 5, agent_B_goalRow = 45, agent_B_goalCol = 45, goalReward = 100, obstReward =-500, stayReward = -5, gamma = 0.9):

		self.maxRow = maxRow
		self.maxCol = maxCol
		self.num_obstacle_pts = num_obstacle_pts
		self.cor_pr = cor_pr
		self.wr_pr = wr_pr
		self.n_actions = n_actions
		self.agent_A_startRow = agent_A_startRow
		self.agent_A_startCol = agent_A_startCol
		self.agent_A_goalRow = agent_A_goalRow
		self.agent_A_goalCol = agent_A_goalCol
		self.agent_B_startRow = agent_B_startRow
		self.agent_B_startCol = agent_B_startCol
		self.agent_B_goalRow = agent_B_goalRow
		self.agent_B_goalCol = agent_B_goalCol		
		self.goalReward = goalReward
		self.obstReward = obstReward
		self.stayReward = stayReward
		self.gamma = gamma
		self.n_agents = 2

		#Initializing model outputs
		self.not_occupied = 0
		self.oRow = []
		self.oCol = []
		self.m = np.zeros((2, self.maxRow, self.maxCol))
		self.st = np.zeros((self.n_actions, self.maxRow*self.maxCol, self.maxRow*self.maxCol))
		self.num_states = self.maxRow*self.maxCol
		self.rm = np.ones((self.num_states, self.n_actions))*self.stayReward


	def rotate(self, l, n):

		"""
		rotate a list n times in anticlockwise direction
		"""
		return l[n:] + l[:n]

	def get_obstacles(self):
		"""
		returns a list of obstacle state indexes for rows and columns

		"""

		#Getting boundaries

		#Top wall
		row = 0
		for col in range(self.maxCol):
			self.oRow.append(row)
			self.oCol.append(col)

		#Bottom wall
		row = self.maxRow - 1
		for col in range(self.maxCol):
			self.oRow.append(row)
			self.oCol.append(col)

		#Left side wall
		col = 0
		for row in range(1, self.maxRow-1):
			self.oRow.append(row)
			self.oCol.append(col)

		#Right side wall
		col = self.maxCol - 1
		for row in range(1, self.maxRow -1):
			self.oRow.append(row)
			self.oCol.append(col)
		wallRows = [2,3,1,3,3]
		wallCols = [4,3,2,2,3]

		self.oRow.extend(wallRows)
		self.oCol.extend(wallCols)
		# print(self.oRow)
		# print(self.oCol)
		return None

	def build_map(self):

		"""
		builds the map using obstacle state indexes
		"""

		cur_state = 0
		for row in range(self.maxRow):
			for col in range(self.maxCol):
				self.m[0][row][col] = cur_state
				cur_state+=1

		for row, col in zip(self.oRow, self.oCol):
			if(self.not_occupied == 1):

				self.m[1][row][col] = 0
			else:
				self.m[1][row][col] = 1 
				
		return None

	def build_st_trans_matrix(self):
		"""
		Function that builds state transition model for input to the MDP

		"""
		if((self.cor_pr + (self.n_actions - 1)*self.wr_pr) < 0.99):
			raise ValueError ('Sum of probabilities dont match')

		if(self.not_occupied!=0):
			if(self.not_occupied!=1):
				raise ValueError ('not occupied should be either zero or one')

		#Actions should start from right and go in anti clockwise direction

		act_map = []
		cur_states_a1 = []
		for row in range(self.maxRow):
			for col in range(self.maxCol):

				cur_state = self.m[0][row][col]
				cur_occ = self.m[1][row][col]

				if(cur_occ == self.not_occupied):
					
					right_state = self.m[0][row][col + 1]
					right_occ = self.m[1][row][col + 1]

					top_state = self.m[0][row - 1][col]
					top_occ = self.m[1][row - 1][col]

					left_state = self.m[0][row][col - 1]
					left_occ = self.m[1][row][col - 1]

					bottom_state = self.m[0][row + 1][col]
					bottom_occ = self.m[1][row + 1][col]

					action_map = [right_state, top_state, left_state, bottom_state]
					occ_map = [right_occ, top_occ, left_occ, bottom_occ]

					
					for action in range(self.n_actions):

						occ_map_rot = self.rotate(occ_map, action)
						action_map_rot = self.rotate(action_map, action)

						prob_sum = 0
						for inner_action in range(self.n_actions):
							
							#assign the probability of correct prob to the state in the direction of action
							if(inner_action == 0):

								self.st[action][int(cur_state)][int(action_map_rot[inner_action])] = self.cor_pr
								prob_sum+=self.cor_pr

							elif(inner_action !=2):

								self.st[action][int(cur_state)][int(action_map_rot[inner_action])] = self.wr_pr
								prob_sum+=self.wr_pr


		
		for action in range(self.n_actions):

			rowSum = np.sum(self.st[action][:][:], axis = 1)
			zeroInd = np.where(rowSum == 0)[0]
			lessInd = np.where(rowSum < 1)[0]

			#Assigning probability to unreachable states so that sum of each row of st matrix becomes 1	
			for row in zeroInd:
				col = random.choice(range(0, self.num_states -1))
				self.st[action][row][col] = 1

			#If the sum of probability for each row is not one assigning the 1 - total probability to some random state
			for row in lessInd:
				col = random.choice(range(0, self.num_states - 1))
				self.st[action][row][col] += 1 - np.sum(self.st[action][row][:])

		return None

	def build_reward_matrix(self,agent):

		"""
		Function that builds the reward model of the world for input to the MDP for path planning

		"""
		right_action = 0
		top_action = 1
		left_action = 2
		bottom_action = 3

		
		if agent ==1:
			goal_state = self.m[0][self.agent_A_goalRow][self.agent_A_goalCol]
		else:
			goal_state = self.m[0][self.agent_B_goalRow][self.agent_A_goalCol]

		
		for row in range(self.maxRow):

			for col in range(self.maxCol):

				cur_occ = self.m[1][row][col]
				if (cur_occ == self.not_occupied):

					cur_state = int(self.m[0][row][col])
					right_state = int(self.m[0][row][col + 1])
					top_state = int(self.m[0][row - 1][col])
					left_state = int(self.m[0][row][col - 1])
					bottom_state = int(self.m[0][row + 1][col])

					right_occ = int(self.m[1][row][col + 1])
					top_occ = int(self.m[1][row - 1][col])
					left_occ = int(self.m[1][row][col - 1])
					bottom_occ = int(self.m[1][row + 1][col])

					if(right_occ != self.not_occupied):
						self.rm[cur_state][right_action] = self.obstReward

					if(right_state == goal_state):
						self.rm[cur_state][right_action] = self.goalReward

					if(top_occ != self.not_occupied):
						self.rm[cur_state][top_action] = self.obstReward

					if(top_state == goal_state):
						self.rm[cur_state][top_action] = self.goalReward

					if(left_occ != self.not_occupied):
						self.rm[cur_state][left_action] = self.obstReward

					if(left_state == goal_state):
						self.rm[cur_state][left_action] = self.goalReward

					if(bottom_occ != self.not_occupied):
						self.rm[cur_state][bottom_action] = self.obstReward

					if(bottom_state == goal_state):
						self.rm[cur_state][bottom_action] = self.goalReward

		return None

def get_reward(startRow, startCol, goalRow, goalCol, oCol, oRow, num_states, m, optimal_policy, rm):

	cur_row = startRow
	cur_col = startCol

	opt_row = [startRow]
	opt_col = [startCol]
	max_points = num_states
	cur_point = 0

	# print('current state  row: {}, col: {}'.format(self.startRow, self.startCol))
	total_reward = 0
	while(1):

		cur_state = int(m[0][cur_row][cur_col])
		cur_opt_action = optimal_policy[str(cur_state)]
		total_reward += rm[cur_state][cur_opt_action]

		if(cur_opt_action == 0):
			cur_row = cur_row
			cur_col = cur_col + 1
		elif(cur_opt_action == 1):
			cur_row = cur_row - 1
			cur_col = cur_col
		elif(cur_opt_action == 2):
			cur_row = cur_row
			cur_col = cur_col - 1
		else:
			cur_row = cur_row + 1
			cur_col = cur_col

		#Printing optimal action
		# print('Action: {}'.format(cur_opt_action))
		# print('Optimal Utility value for this state and this action : {}'.format(self.expected_values[int(cur_state)]))
		# print('Transition probability for current state and current action : {}'.format(self.st[cur_opt_action][int(cur_state)][int(self.m[0][cur_row][cur_col])]))
		# print('Reward for current state to perform current action : {}'.format(self.rm[int(cur_state)][cur_opt_action]))


		opt_row.append(cur_row)
		opt_col.append(cur_col)

		
		cur_point+=1

		if(cur_row == goalRow):
			if(cur_col == goalCol):
				print('Goal Reached!!')
				break

		if(cur_point == max_points):
			print('Steps limit over!!')
			break

	return total_reward

def visualize_path(startRow, startCol, goalRow, goalCol, oCol, oRow, num_states, m, optimal_policy, rm,algorithm):

	#Visualize path
	cur_row = startRow
	cur_col = startCol

	opt_row = [startRow]
	opt_col = [startCol]
	max_points = num_states
	cur_point = 0
	agent_policy = []
	# print('current state  row: {}, col: {}'.format(self.startRow, self.startCol))
	while(1):

		cur_state = int(m[0][cur_row][cur_col])
		cur_opt_action = optimal_policy[str(cur_state)]
		# print("TESTING 0210")
		# print(cur_state)
		# print(cur_opt_action)
		agent_policy.append((cur_state,cur_opt_action))
		# print(agent_policy)

		if(cur_opt_action == 0):
			cur_row = cur_row
			cur_col = cur_col + 1
		elif(cur_opt_action == 1):
			cur_row = cur_row - 1
			cur_col = cur_col
		elif(cur_opt_action == 2):
			cur_row = cur_row
			cur_col = cur_col - 1
		else:
			cur_row = cur_row + 1
			cur_col = cur_col

		#Printing optimal action
		# print('Action: {}'.format(cur_opt_action))
		# print('Optimal Utility value for this state and this action : {}'.format(self.expected_values[int(cur_state)]))
		# print('Transition probability for current state and current action : {}'.format(self.st[cur_opt_action][int(cur_state)][int(self.m[0][cur_row][cur_col])]))
		# print('Reward for current state to perform current action : {}'.format(self.rm[int(cur_state)][cur_opt_action]))


		opt_row.append(cur_row)
		opt_col.append(cur_col)

		# print('current State row : {}, col : {}'.format(opt_row[-1], opt_col[-1]))

		# ax.plot(opt_col, opt_row, linewidth = 5, color = 'red')
		# plt.pause(0.1)

		cur_point+=1

		if(cur_row == goalRow):
			if(cur_col == goalCol):
				print('Goal Reached!!')
				return agent_policy
				break

		if(cur_point == max_points):
			print('Steps limit over!!')
			break


def visualize_policy(maxRow, maxCol, startRow, startCol, goalRow, goalCol, oCol, oRow, num_states, m, optimal_policy, rm, algorithm):

	#Visualize world
	fig, ax = plt.subplots(figsize = (5.8,5.8))
	plt.ion()
	ax.scatter(oCol, oRow, marker = 's',s = 700, c = 'black')
	ax.scatter(startCol, startRow, s = 700, c = 'b')
	ax.scatter(goalCol, goalRow, s = 700,c = 'g')
	plt.axis("equal")
	plt.axis('tight')
	


	#Arrow config
	arrow_head_len = 0.1
	len_arrow= 1-2*arrow_head_len
	arrow = {}
	arrow['right']={'sx':-1*(len_arrow/2), 'sy':0 ,'dx':len_arrow, 'dy':0}
	arrow['top']={'sx':0, 'sy':(len_arrow/2) ,'dx':0, 'dy':-1*len_arrow}
	arrow['left']={'sx': (len_arrow/2), 'sy':0 ,'dx':-1*len_arrow, 'dy':0}
	arrow['bottom']={'sx': 0, 'sy':-1*(len_arrow/2) ,'dx':0, 'dy':len_arrow}


	#Visualize path
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



			if(cur_opt_action == 0):
				direction = 'right'
				x = arrow[direction]['sx']+cur_col
				y = arrow[direction]['sy']+cur_row
				dx = arrow[direction]['dx']
				dy = arrow[direction]['dy']

			elif(cur_opt_action == 1):
				direction = 'top'
				x = arrow[direction]['sx']+cur_col
				y = arrow[direction]['sy']+cur_row
				dx = arrow[direction]['dx']
				dy = arrow[direction]['dy']

			elif(cur_opt_action == 2):
				direction = 'left'
				x = arrow[direction]['sx']+cur_col
				y = arrow[direction]['sy']+cur_row
				dx = arrow[direction]['dx']
				dy = arrow[direction]['dy']
			
			else:
				direction = 'bottom'	
				x = arrow[direction]['sx'] + cur_col
				y = arrow[direction]['sy'] + cur_row
				dx = arrow[direction]['dx']
				dy = arrow[direction]['dy']

			ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.1, fc='k', ec='k')

	figname = 'policy_'+algorithm
	fig.savefig(figname)
	plt.close()

def fit_policy(st, rm, gamma, num_states):
	"""
	This function trains an optimal policy using Markov Decision Process using MDPToolbox
	using PolicyIteration

	"""
	iterations = list(range(1,1000,10))
	data_policy = {}
	data_policy['convergence'] = {}

	for iter in iterations:

		print('Current Iteration: {}'.format(iter))

		data_policy[str(iter)] = {}

		tot_time_start = time.time()
		vi = mdptoolbox.mdp.PolicyIteration(st, rm, gamma, max_iter = 10000000, eval_type = 1)
		# vi.setVerbose()
		time_iter, iter_value, iter_policy, policy_change, policies = vi.run(max_iter = iter)
		tot_time_end = time.time()
		tot_time = tot_time_end - tot_time_start

		policy_change = [int(x) for x in policy_change]
		if(np.any(np.array(iter_value) > iter)):
			raise ValueError('Value loop of Policy Iteration not stopping at maximum iterations provided')


		data_policy[str(iter)]['tot_time'] = tot_time
		data_policy[str(iter)]['time_iter'] = time_iter
		data_policy[str(iter)]['policy_iter'] = iter_policy
		data_policy[str(iter)]['value_iter'] = iter_value
		data_policy[str(iter)]['policy_change'] = policy_change

		

	print('Convergence')
	tot_time_start = time.time()
	vi = mdptoolbox.mdp.PolicyIteration(st, rm, gamma, max_iter = 10000000, eval_type = 1)
	time_iter, iter_value, iter_policy_policy, policy_change, policies = vi.run(max_iter = 10000)
	tot_time_end = time.time()

	policy_change = [int(x) for x in policy_change]
	policies = [tuple(int(x) for x in opt_policy) for opt_policy in policies]
	optimal_policy = vi.policy
	expected_values = vi.V
	optimal_policy = tuple(int(x) for x in optimal_policy)
	expected_values = tuple(float(x) for x in expected_values)

	optimal_policy = dict(zip(list(range(num_states)), list(optimal_policy)))
	expected_values = list(expected_values)
	policies = [dict(zip(list(range(num_states)), list(opt_policy))) for opt_policy in policies]


	data_policy['convergence']['tot_time'] = tot_time_end - tot_time_start
	data_policy['convergence']['time_iter'] = time_iter
	data_policy['convergence']['policy_iter'] = iter_policy_policy
	data_policy['convergence']['value_iter'] = iter_value
	data_policy['convergence']['policy_change'] = policy_change
	data_policy['convergence']['optimal_policy'] = optimal_policy
	data_policy['convergence']['expected_values'] = expected_values
	data_policy['convergence']['policies'] = policies

	return data_policy,optimal_policy

def store_to_file(data, file_name):

	with open(file_name, 'w') as outfile:
		json.dump(data, outfile)

def fit_value(st, rm, gamma, num_states):
	"""
	This function trains an optimal policy using Markov Decision Process using MDPToolbox
	using ValueIteration

	"""
	iterations = list(range(1,1000,10))
	data_value = {}
	data_value['convergence'] = {}
	for iter in iterations:

		print('Current Iteration: {}'.format(iter))
		data_value[str(iter)] = {}

		tot_time_start = time.time()
		vi = mdptoolbox.mdp.ValueIteration(st, rm, gamma, max_iter = 10000000, epsilon = 0.0001)
		# vi.setVerbose()
		time_iter, iter_value, variation, policies = vi.run(max_iter = iter)
		tot_time_end = time.time()
		tot_time = tot_time_end - tot_time_start

		if(iter_value > iter):
			raise ValueError('ValueIteration is not stopping at maximum iterations')

		data_value[str(iter)]['tot_time'] = tot_time
		data_value[str(iter)]['time_iter'] = time_iter
		data_value[str(iter)]['value_iter'] = iter_value
		data_value[str(iter)]['variation'] = variation



	print('Convergence')
	tot_time_start = time.time()
	vi = mdptoolbox.mdp.ValueIteration(st, rm, gamma, max_iter = 10000, epsilon = 0.0001)
	time_iter, iter_value, variation, policies = vi.run(max_iter = 10000)
	tot_time_end = time.time()

	optimal_policy = vi.policy
	expected_values = vi.V
	policies = [tuple(int(x) for x in opt_policy) for opt_policy in policies]
	optimal_policy = tuple(int(x) for x in optimal_policy)
	expected_values = tuple(float(x) for x in expected_values)

	optimal_policy = dict(zip(list(range(num_states)), list(optimal_policy)))
	expected_values = list(expected_values)
	policies = [dict(zip(list(range(num_states)), list(opt_policy))) for opt_policy in policies]

	
	data_value['convergence']['tot_time'] = tot_time_end - tot_time_start
	data_value['convergence']['time_iter'] = time_iter
	data_value['convergence']['value_iter'] = iter_value
	data_value['convergence']['variation'] = variation
	data_value['convergence']['optimal_policy'] = optimal_policy
	data_value['convergence']['expected_values'] = expected_values
	data_value['convergence']['policies'] = policies

	return data_value

def plot_analysis(file_data_world, file_data_value, file_data_policy, file_optimal_policy,agent):

	iterations = list(range(1,1000,10))
	with open(file_data_world) as json_data:
		data_world = json.load(json_data)

	with open(file_data_value) as json_data:
		data_value = json.load(json_data)

	with open(file_data_policy) as json_data:
		data_policy = json.load(json_data)

	with open(file_optimal_policy) as json_data:
		optimal_policy = json.load(json_data)


	#Total computation time
	tot_time_policy = []
	tot_time_value = []
	for iter in iterations:
		tot_time_policy.append(data_policy[str(iter)]['tot_time'])
		tot_time_value.append(data_value[str(iter)]['tot_time'])


	#Average value update time
	value_time_policy = []
	value_time_value = []

	for itr in iterations:

		time_policy = []
		for time_value in data_policy[str(itr)]['time_iter']:
			time_policy.append(sum(time_value)/float(len(time_value)))

		value_time_policy.append(sum(time_policy)/len(time_policy))
		value_time_value.append(sum(data_value[str(itr)]['time_iter'])/len(data_value[str(itr)]['time_iter']))



	#Visualize path and policy for policy iteration and value iteration
	if agent == 1:
		agent_policy = visualize_path(data_world['agent_A_startRow'], data_world['agent_A_startCol'], data_world['agent_A_goalRow'], data_world['agent_A_goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'], data_policy['convergence']['optimal_policy'], data_world['rm'],'policy_iteration')
	else:
		agent_policy = visualize_path(data_world['agent_B_startRow'], data_world['agent_B_startCol'], data_world['agent_B_goalRow'], data_world['agent_B_goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'], data_policy['convergence']['optimal_policy'], data_world['rm'],'policy_iteration')

	#Calculating reward
	policy_policy = data_policy['convergence']['policies']
	policy_value = data_value['convergence']['policies']

	print(len(policy_policy))
	print(len(policy_value))
	reward_policy = []
	reward_value = []
	for p_pol in policy_policy:
		if agent == 1:
			reward_p = get_reward(data_world['agent_A_startRow'], data_world['agent_A_startCol'], data_world['agent_A_goalRow'], data_world['agent_A_goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'],p_pol , data_world['rm'])
			reward_policy.append(reward_p)
		else: 
			reward_p = get_reward(data_world['agent_B_startRow'], data_world['agent_B_startCol'], data_world['agent_B_goalRow'], data_world['agent_B_goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'],p_pol , data_world['rm'])
			reward_policy.append(reward_p)

	for v_pol in policy_value:
		if agent == 1:
			reward_v = get_reward(data_world['agent_A_startRow'], data_world['agent_A_startCol'], data_world['agent_A_goalRow'], data_world['agent_A_goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'],v_pol , data_world['rm'])
			reward_value.append(reward_v)
		else:
			reward_v = get_reward(data_world['agent_B_startRow'], data_world['agent_B_startCol'], data_world['agent_B_goalRow'], data_world['agent_B_goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'],v_pol , data_world['rm'])
			reward_value.append(reward_v)

	print(agent_policy)
	return agent_policy
	


class AgentState(PathPlanning):
	def __init__(self, maxRow, maxCol, num_obstacle_pts, cor_pr, wr_pr, n_actions, startRow, startCol, goalRow, goalCol, goalReward, obstReward, stayReward,  gamma):
		
		self.agents=agents 
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
		self.goalReward = goalReward
		self.obstReward = obstReward
		self.stayReward = stayReward
		self.gamma = gamma
		self.n_agents = 2

		#Initializing model outputs
		self.not_occupied = 0
		self.oRow = []
		self.oCol = []
		self.m = np.zeros((2, self.maxRow, self.maxCol),dtype=int)
		self.st = np.zeros((self.n_actions, self.maxRow*self.maxCol, self.maxRow*self.maxCol))
		self.num_states = self.maxRow*self.maxCol
		self.rm = np.ones((self.num_states, self.n_actions))*self.stayReward


	def rotate(self, l, n):

		"""
		rotate a list n times in anticlockwise direction
		"""
		return l[n:] + l[:n]

	def get_obstacles(self):
		"""
		returns a list of obstacle state indexes for rows and columns

		"""

		#Getting boundaries

		#Top wall
		row = 0
		for col in range(self.maxCol):
			self.oRow.append(row)
			self.oCol.append(col)

		#Bottom wall
		row = self.maxRow - 1
		for col in range(self.maxCol):
			self.oRow.append(row)
			self.oCol.append(col)

		#Left side wall
		col = 0
		for row in range(1, self.maxRow-1):
			self.oRow.append(row)
			self.oCol.append(col)

		#Right side wall
		col = self.maxCol - 1
		for row in range(1, self.maxRow -1):
			self.oRow.append(row)
			self.oCol.append(col)
		wallRows = [2,3,1,3,3]
		wallCols = [4,3,2,2,3]

		self.oRow.extend(wallRows)
		self.oCol.extend(wallCols)
		# print(self.oRow)
		# print(self.oCol)
		return None

	def build_map(self):

		"""
		builds the map using obstacle state indexes
		"""

		cur_state = 0
		for row in range(self.maxRow):
			for col in range(self.maxCol):
				self.m[0][row][col] = cur_state
				cur_state+=1

		for row, col in zip(self.oRow, self.oCol):
			if(self.not_occupied == 1):

				self.m[1][row][col] = 0
			else:
				self.m[1][row][col] = 1 
				
		return None

	def build_st_trans_matrix(self):
		"""
		Function that builds state transition model for input to the MDP

		"""
		if((self.cor_pr + (self.n_actions - 1)*self.wr_pr) < 0.99):
			raise ValueError ('Sum of probabilities dont match')

		if(self.not_occupied!=0):
			if(self.not_occupied!=1):
				raise ValueError ('not occupied should be either zero or one')

		#Actions should start from right and go in anti clockwise direction

		act_map = []
		cur_states_a1 = []
		for row in range(self.maxRow):
			for col in range(self.maxCol):

				cur_state = self.m[0][row][col]
				cur_occ = self.m[1][row][col]

				if(cur_occ == self.not_occupied):
					
					right_state = self.m[0][row][col + 1]
					right_occ = self.m[1][row][col + 1]

					top_state = self.m[0][row - 1][col]
					top_occ = self.m[1][row - 1][col]

					left_state = self.m[0][row][col - 1]
					left_occ = self.m[1][row][col - 1]

					bottom_state = self.m[0][row + 1][col]
					bottom_occ = self.m[1][row + 1][col]

					action_map = [right_state, top_state, left_state, bottom_state]
					occ_map = [right_occ, top_occ, left_occ, bottom_occ]

					
					for action in range(self.n_actions):

						occ_map_rot = self.rotate(occ_map, action)
						action_map_rot = self.rotate(action_map, action)

						prob_sum = 0
						for inner_action in range(self.n_actions):
							
							#assign the probability of correct prob to the state in the direction of action
							if(inner_action == 0):

								self.st[action][int(cur_state)][int(action_map_rot[inner_action])] = self.cor_pr
								prob_sum+=self.cor_pr

							elif(inner_action !=2):

								self.st[action][int(cur_state)][int(action_map_rot[inner_action])] = self.wr_pr
								prob_sum+=self.wr_pr


		
		for action in range(self.n_actions):

			rowSum = np.sum(self.st[action][:][:], axis = 1)
			zeroInd = np.where(rowSum == 0)[0]
			lessInd = np.where(rowSum < 1)[0]

			#Assigning probability to unreachable states so that sum of each row of st matrix becomes 1	
			for row in zeroInd:
				col = random.choice(range(0, self.num_states -1))
				self.st[action][row][col] = 1

			#If the sum of probability for each row is not one assigning the 1 - total probability to some random state
			for row in lessInd:
				col = random.choice(range(0, self.num_states - 1))
				self.st[action][row][col] += 1 - np.sum(self.st[action][row][:])

		return None

	def build_reward_matrix(self):

		"""
		Function that builds the reward model of the world for input to the MDP for path planning

		"""
		right_action = 0
		top_action = 1
		left_action = 2
		bottom_action = 3

		
		goal_state = self.m[0][self.goalRow][self.goalCol]

		
		for row in range(self.maxRow):

			for col in range(self.maxCol):

				cur_occ = self.m[1][row][col]
				if (cur_occ == self.not_occupied):

					cur_state = int(self.m[0][row][col])
					right_state = int(self.m[0][row][col + 1])
					top_state = int(self.m[0][row - 1][col])
					left_state = int(self.m[0][row][col - 1])
					bottom_state = int(self.m[0][row + 1][col])

					right_occ = int(self.m[1][row][col + 1])
					top_occ = int(self.m[1][row - 1][col])
					left_occ = int(self.m[1][row][col - 1])
					bottom_occ = int(self.m[1][row + 1][col])

					if(right_occ != self.not_occupied):
						self.rm[cur_state][right_action] = self.obstReward

					if(right_state == goal_state):
						self.rm[cur_state][right_action] = self.goalReward

					if(top_occ != self.not_occupied):
						self.rm[cur_state][top_action] = self.obstReward

					if(top_state == goal_state):
						self.rm[cur_state][top_action] = self.goalReward

					if(left_occ != self.not_occupied):
						self.rm[cur_state][left_action] = self.obstReward

					if(left_state == goal_state):
						self.rm[cur_state][left_action] = self.goalReward

					if(bottom_occ != self.not_occupied):
						self.rm[cur_state][bottom_action] = self.obstReward

					if(bottom_state == goal_state):
						self.rm[cur_state][bottom_action] = self.goalReward

		return None

	def build_world(self):

		size = 6
		seed=1337

	    # self.get_obstacles()
		self.get_obstacles()
		self.build_map()
		self.build_st_trans_matrix()
		self.build_reward_matrix()
	
		world_data= {}
		world_data['st'] = self.st.tolist()
		world_data['rm'] = self.rm.tolist()
		world_data['gamma'] = self.gamma
		world_data['num_states'] = self.num_states
		world_data['startRow'] = self.startRow
		world_data['startCol'] = self.startCol
		world_data['goalRow'] = self.goalRow
		world_data['goalCol'] = self.goalCol
		world_data['oCol'] = self.oCol
		world_data['oRow'] = self.oRow
		world_data['m'] = self.m.tolist()
		world_data['maxRow'] = self.maxRow
		world_data['maxCol'] = self.maxCol

		world_data['st'] = np.array(world_data['st'])
		world_data['rm'] = np.array(world_data['rm'])
		world_data['m'] = np.array(world_data['m'])

		terminal_states = [(world_data['goalRow'],world_data['goalCol'])]

		P=world_data['st']
		p0 = np.ones(P.shape[1])/P.shape[1]
	
		return(world_data['st'],world_data['rm'],world_data['gamma'],world_data['num_states'],terminal_states,size,p0,0.0001, 10000000,seed)

	def get_initialstate(self):
		self.build_map()
		initial_state = self.m[0][self.startRow][self.startCol]
		return initial_state

	def choose_action(optimal_policy,state):
		# print(type(optimal_policy))
		# print(optimal_policy)
		# print(type(state))
		# print(state)
		action = optimal_policy.get(state)
		# print(action)
		return action

	def get_policy(self):
		world_data['st'], world_data['rm'], world_data['gamma'], world_data['num_states'],terminal_states,size,p0,epsilon, max_iter,seed = self.build_world()
		data_policy,optimal_policy = fit_policy(world_data['st'], world_data['rm'], world_data['gamma'], world_data['num_states'])
		return optimal_policy
	
	def get_mdp(self):
	
		world_data['st'], world_data['rm'], world_data['gamma'], world_data['num_states'],terminal_states,size,p0,epsilon, max_iter,seed = self.build_world()
		print(Env(md.MDP(world_data['st'],world_data['rm'],world_data['gamma'],terminal_states,size,p0,epsilon = 0.0001, max_iter = 10000000,seed=seed)))
		return Env(md.MDP(world_data['st'],world_data['rm'],world_data['gamma'],terminal_states,size,p0,epsilon = 0.0001, max_iter = 10000000,seed=seed))

	def choose(self):
		initial_state=AgentState.get_initialstate(self)
		agent_action = choose_action(initial_state,optimal_policy)
		print(agent_action)

	def choose_action(state,optimal_policy):

		action = optimal_policy.get(state)
		# print(action)
		return action
		# action = optimal_policy.get(state)
		# print(action)
		# return action	
		
class Env(AgentState):
	def __init__(self,agent_mdp):
		self.agent_mdp = agent_mdp	

class PolicyAgent(AgentState):
	def __init__(self,optimal_policy):
		self.optimal_policy = optimal_policy
	

	def __str__(self):
		return str(self.optimal_policy)
	
	def choose_action(state,optimal_policy):

		action = optimal_policy.get(state)
		# print(action)
		return action
		# action = optimal_policy.get(state)
		# print(action)
		# return action


env_params = {
			'maxRow' : 6,
			'maxCol' : 6,
			'num_obstacle_pts' : 10,
			'cor_pr' : 0.95,
			'wr_pr' : 0.025,
			'n_actions' : 4,
			'goalReward' : 1,
			'obstReward' : -15,
			'stayReward' : -0.03,
			'gamma' : 0.98
			}

agents_param ={ 'agent_A' : { 'startRow' : 2, 'startCol' : 2, 'goalRow' : 4, 'goalCol' : 3, },
				'agent_B' : { 'startRow' : 1, 'startCol' : 1, 'goalRow' : 4, 'goalCol' : 1, } 
			  }

def define_env(agent):
	
	world_params = env_params.copy()
	world_params.update(agents_param[agent])
	return AgentState(**world_params)


#run time execution to step
for agent in agents:
	initial_env = define_env
	initial_env.choose()

print(agent_step)



for agent in agents:
	while agent_step[agent][2] == False:
		agent_action[agent] = PolicyAgent.choose_action(agent_step[agent][0],optimal_policy[agent])
		print("Agent action")
		print(agent_action)
		for agent in range(agent_action[agent]):
			agent_step[agent] = agent_mdp[agent].step(agent_action[agent])
			# print("Agent step function result")
			print(agent_step)



