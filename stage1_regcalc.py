import numpy as np
import networkx as nx

np.set_printoptions(edgeitems=30, linewidth = 1000000)
class Graph:
    def __init__(self, vertices):
        self.M = vertices   # Total number of vertices in the graph
        self.graph = []     # Array of edges
    # Add edges
    def add_edge(self, a, b, c):
        self.graph.append([a, b, c])
    # Print the solution
    def print_solution(self, distance, maxRow, maxCol):
        bellman_matrix = np.reshape(distance, (maxRow, maxCol))
        # print("Vertex Distance from Source")
        # print(bellman_matrix)
        bellman_matrix=bellman_matrix.tolist()
        return bellman_matrix
    
    def bellman_ford(self, src, maxRow, maxCol):
        distance = [float("Inf")] * self.M
        distance[src] = 0
        for i in range(self.M - 1):
            for a, b, c in self.graph:
                if distance[a] != float("Inf") and distance[a] + c < distance[b]:
                    distance[b] = distance[a] + c
            # print(f"Iteration {i+1}: {distance}")
        for a, b, c in self.graph:
            if distance[a] != float("Inf") and distance[a] + c < distance[b]:
                # print("Graph contains negative weight cycle")
                return
        return self.print_solution(distance, maxRow, maxCol)
    
def bellmancalc(cellvaluematrix):
    
    maxRow, maxCol = cellvaluematrix.shape
    m = np.arange(0, maxRow * maxCol).reshape(maxRow, maxCol)
    # print(m)
    last_elements = m[:, -1]
    first_elements = m[:, 0]

    # Find the walls where cellvaluematrix is 100
    indices = np.where(cellvaluematrix == 100)

    # Use these indices to access the corresponding values in m
    walls = m[indices]

    action_map=[]
    g = Graph(maxRow*maxCol)
    for row in range(maxRow):
        for col in range(maxCol):
            action_map=[]
            cur_state = m[row][col]
            x = last_elements
            y = first_elements
            if cur_state not in x:
                right_state = m[row][col + 1]
                action_map.append(right_state)
            if row > 0:
                top_state = m[row - 1][col]
                action_map.append(top_state)
            if cur_state not in y:
                left_state = m[row][col - 1]
                action_map.append(left_state)
            if row < maxRow - 1:
                bottom_state = m[row + 1][col]
                action_map.append(bottom_state)
            for nextstate in action_map:
                # print("u,v,w", cur_state,action_map)
                if nextstate in walls:
                    weight = 100 
                else:
                    weight = 1
                # print("cur_state,nextstate,weight",cur_state,nextstate,weight)
                g.add_edge(cur_state,nextstate,weight)
    return g

# calculate regret
def regret(agA_startrow, agA_startcol, cellvalue_agA, bellman_agA, agB_startrow, agB_startcol, cellvalue_agB, bellman_agB):
    
    regret_agA = cellvalue_agA.copy()
    # print("cellvalue_agA[agA_startrow][agA_startcol]",cellvalue_agA[agA_startrow][agA_startcol])
    # print("agA_startrow,agA_startrow",agA_startrow,agA_startcol)
    # print("cellvalue_agB[agB_startrow][agB_startcol]",cellvalue_agB[agB_startrow][agB_startcol])

    # print("bellman_agA",bellman_agA)
    # print("\ncellvalue_agA",cellvalue_agA)
    # print("\nbellman_agB",bellman_agB)
    # print("\ncellvalue_agB",cellvalue_agB)

    for i in range(len(regret_agA)):
        for j in range(len(regret_agA[0])):
            regret_agA[i][j] = bellman_agA[i][j] + cellvalue_agA[i][j] - cellvalue_agA[agA_startrow][agA_startcol]

    # AGENT B CALCULATION
    regret_agB = cellvalue_agB.copy()
    for i in range(len(regret_agB)):
        for j in range(len(regret_agB[0])):
            regret_agB[i][j] = bellman_agB[i][j] + cellvalue_agB[i][j] - cellvalue_agB[agB_startrow][agB_startcol]

    # print("***********************************************************************")
    # print("REGRET_AGENT_A\n")
    # print(regret_agA)
    # print("***********************************************************************")
    # print("REGRET_AGENT_B\n")
    # print(regret_agB)
    return (regret_agA, regret_agB)


def reg_calc(agA_startrow, agA_startcol, startcellA, cellvalue_agA, agB_startrow, agB_startcol, startcellB, cellvalue_agB):
    # print("VAL IN REG_CALC")
    # print("agA_startrow, agA_startcol,",agA_startrow, agA_startcol)
    # print("agB_startrow, agB_startcol",agB_startrow, agB_startcol)
    # print(cellvalue_agA)
    # print(cellvalue_agB)
    maxRow, maxCol = cellvalue_agA.shape
    m = np.arange(0, maxRow * maxCol).reshape(maxRow, maxCol)

    if maxCol==maxRow:
        srcA = (agA_startrow * maxCol + agA_startcol)
        srcB = (agB_startrow * maxCol + agB_startcol)
    else:
        srcA = (agA_startrow * maxCol + agA_startcol)
        srcB = (agB_startrow * maxCol + agB_startcol)
   

    g = bellmancalc(cellvalue_agA)
    # print("**********AGENT_A******************")
    bellman_agA = g.bellman_ford(srcA,maxRow, maxCol) #startcell

    print("**********AGENT_B******************")
    bellman_agB = g.bellman_ford(srcB,maxRow, maxCol)

    if maxCol==maxRow:
        regret_landscapeA,regret_landscapeB = regret(agA_startrow, agA_startcol, cellvalue_agA, bellman_agA, agB_startrow, agB_startcol, cellvalue_agB, bellman_agB)
    else:
        adjust_agA_startcol= agA_startcol
        adjust_agB_startcol= agB_startcol
        regret_landscapeA,regret_landscapeB = regret(agA_startrow, adjust_agA_startcol, cellvalue_agA, bellman_agA, agB_startrow, adjust_agB_startcol, cellvalue_agB, bellman_agB)

    
    return regret_landscapeA,regret_landscapeB


def find_closed_loop_L(matrix):
    rows, cols = matrix.shape

    G = nx.Graph()

    # Build graph using only L cells
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 'L':
                G.add_node((i, j))

    # Connect adjacent L cells
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 'L':
                for di, dj in [(1,0), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if matrix[ni][nj] == 'L':
                            G.add_edge((i, j), (ni, nj))

    # Find L cycles
    cycles = nx.cycle_basis(G)

    # Keep only cycles that enclose H
    for cycle in cycles:
        if has_enclosed_H(matrix, set(cycle)):
            return True

    return False


def has_enclosed_H(matrix, cycle_cells):
    rows, cols = matrix.shape
    blocked = set(cycle_cells)

    visited = set()
    stack = []

    # Start flood-fill from outside border H/O cells
    for i in range(rows):
        for j in [0, cols - 1]:
            if (i, j) not in blocked:
                stack.append((i, j))

    for j in range(cols):
        for i in [0, rows - 1]:
            if (i, j) not in blocked:
                stack.append((i, j))

    while stack:
        i, j = stack.pop()

        if (i, j) in visited or (i, j) in blocked:
            continue

        visited.add((i, j))

        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj

            if 0 <= ni < rows and 0 <= nj < cols:
                if (ni, nj) not in blocked and (ni, nj) not in visited:
                    stack.append((ni, nj))

    # Any H not reachable from outside is enclosed
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 'H' and (i, j) not in visited and (i, j) not in blocked:
                return True

    return False

def apply_rules(matrix):
    matrix = np.array(matrix).copy()
    rows, cols = matrix.shape

    if find_closed_loop_L(matrix):
        matrix[matrix == 'M'] = 'H'
        print(matrix)
        return matrix

    G = nx.Graph()

    # Add L/M cells as graph nodes
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] in ['L', 'M']:
                G.add_node((i, j))

    # Connect neighbouring L/M cells
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] in ['L', 'M']:
                for di, dj in [(1,0), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if matrix[ni][nj] in ['L', 'M']:
                            G.add_edge((i, j), (ni, nj))

    cycles = nx.cycle_basis(G)

    keep_M = set()

    for cycle in cycles:
        cycle_set = set(cycle)

        if has_enclosed_H(matrix, cycle_set):
            for cell in cycle_set:
                i, j = cell
                if matrix[i][j] == 'M':
                    keep_M.add(cell)

    # Replace M cells not part of an enclosing L/M cycle
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 'M' and (i, j) not in keep_M:
                matrix[i][j] = 'H'

    print("APPLY RULES RESULT MATRIX\n",matrix)
    return matrix

def modify_matrix(matrix):
        print("Matrix in modify_matrix\n",matrix)
        lowest_value = np.min(matrix)
        second_lowest_value = np.min(np.where(matrix == lowest_value, np.inf, matrix))
        regret_agA_replaced = np.where(matrix == lowest_value, 'L', matrix.astype(str))
        regret_agA_replaced = np.where(regret_agA_replaced == str(second_lowest_value), 'M', regret_agA_replaced)
        regret_agA_replaced = np.where((regret_agA_replaced != 'L') & (regret_agA_replaced != 'M'), 'H', regret_agA_replaced)
        print(regret_agA_replaced)
        # print(regret_value_map)
        return regret_agA_replaced



def get_contour_cells(threshold_matrix):
    contour_cells = {}

    for label in ["L", "M", "H"]:
        coords = np.where(threshold_matrix == label)
        cells = set(zip(coords[0], coords[1]))
        contour_cells[label] = cells
        print("threshold_matrix",threshold_matrix)
        print("contour_cells",contour_cells)

    return contour_cells


def calculate_regret_value_map(regret_landscape, threshold_matrix):
    regret_value_map = {}

    if np.any(threshold_matrix == 'L'):
        regret_value_map['L'] = float(np.min(regret_landscape[threshold_matrix == 'L']))

    if np.any(threshold_matrix == 'M'):
        regret_value_map['M'] = float(np.min(regret_landscape[threshold_matrix == 'M']))

    print(regret_value_map)
    return regret_value_map

def threshold_reg(regret_landscapeA, regret_landscapeB):

    print(regret_landscapeA, regret_landscapeB)

    temp_threshold_regA = modify_matrix(regret_landscapeA)
    temp_threshold_regB = modify_matrix(regret_landscapeB)

    threshold_regA = apply_rules(temp_threshold_regA)
    threshold_regB = apply_rules(temp_threshold_regB)

    regret_value_map_A = calculate_regret_value_map(regret_landscapeA, threshold_regA)
    regret_value_map_B = calculate_regret_value_map(regret_landscapeB, threshold_regB)

    contour_cells_A = get_contour_cells(threshold_regA)
    contour_cells_B = get_contour_cells(threshold_regB)



    print("threshold_regA:")
    print(threshold_regA)
    print("regret_value_map_A:")
    print(regret_value_map_A)
    print("contour_cells_A:")
    print(contour_cells_A)

    print("threshold_regB:")
    print(threshold_regB)
    print("regret_value_map_B:")
    print(regret_value_map_B)
    print("contour_cells_B:")
    print(contour_cells_B)

    return threshold_regA, threshold_regB, regret_value_map_A, regret_value_map_B, contour_cells_A, contour_cells_B

