
import copy
import operator
import sys
from enum import Enum

import numpy as np
from Agentsearch import *
from gym import utils
from six import StringIO, b

MOVE_COST = 1.0
# STAY_COST = 0.5
EXIT_COST = 0
COSTS = {
    "up": MOVE_COST,
    "down": MOVE_COST,
    "left": MOVE_COST,
    "right": MOVE_COST,
    # "stay": STAY_COST,
}

OFFGRIDLOC = None
aPos_flag = False
bPos_flag = False

NOACTION = {"noaction": MOVE_COST}

AGOFFGRID = {"agoffgrid": EXIT_COST}


class AgentGridState(InformedProblemState):
    def __init__(self, grid, agAValue_Matrix, agBValue_Matrix):
        self.grid = grid
        self.agAValue_Matrix = agAValue_Matrix
        self.agBValue_Matrix = agBValue_Matrix

    def __str__(self):
        return str(self.grid)

    def equals(self, state):

        return (self.grid == state.grid).all()

    def agentingrid(self, state, a):
        nparray_contains = np.frompyfunc(operator.contains, 2, 1)
        return np.any(nparray_contains(self.grid, a))


    def getAgPosVal(self, agent_Coords, agvalue):
        self.agent_Coords = agent_Coords
        self.agvalue = agvalue
        ag_curr_value = agvalue[agent_Coords]
        return ag_curr_value

    def getOldPos(self, grid, agID):

        nparray_contains = np.frompyfunc(operator.contains, 2, 1)
        ag_Idx = np.argwhere(nparray_contains(grid, agID))
        ag_Coords = ag_Idx.tolist()
        oldPos = tuple(ag_Coords[0])
        return oldPos

    def getnewpos(self, agValue_Matrix):
        newPos = []
        agMax_Value = np.max(agValue_Matrix)
        ag_Idx = np.argwhere(agValue_Matrix == agMax_Value)
        ag_Coords = ag_Idx.tolist()
        for coords in ag_Coords:
            newPos.append(tuple(coords))
        return newPos[0]

    def definegoal(self,goal,agAValue_Matrix, agBValue_Matrix):
        gridcopy = copy.deepcopy(self.grid)
        aPos = self.getOldPos(gridcopy, "A")
        bPos = self.getOldPos(gridcopy, "B")

        gridcopy[aPos] = {"_"}
        gridcopy[bPos] = {"_"}

        return AgentGridState(gridcopy, self.agAValue_Matrix, self.agBValue_Matrix)
    def opt_vector(self, state, pattern):
        
        if pattern == 'parallel':
            # print("parallel")
            A_optcost = 6.0
            B_optcost = 6.0
            Sum_optcost = 12.0
        elif pattern == 'extended_parallel':
            # print("Extended_parallel")
            A_optcost = 7.0
            B_optcost = 7.0
            Sum_optcost = 14.0
        elif pattern == 'zigzag':
            # print("zigzag")
            A_optcost = 7.0
            B_optcost = 7.0
            Sum_optcost = 14.0
        elif pattern == 'around_block':
            # print("around_block")
            A_optcost = 4.0
            B_optcost = 4.0
            Sum_optcost = 8.0
        else:
            raise ValueError(f"Invalid pattern: {pattern}")

        # opt_vector = [A_optcost, B_optcost, max(A_optcost, B_optcost)]
        opt_vector = [A_optcost, B_optcost, Sum_optcost]
        return opt_vector
    def possibleActions(self, state):

        statecopy = copy.deepcopy(state.grid)
        possAActions = []
        agentingrid = self.agentingrid(statecopy, "A")
        if agentingrid:
            agApos = self.getOldPos(statecopy, "A")
            goalApos = self.getnewpos(self.agAValue_Matrix)
            if agApos == goalApos:
                # aPos_flag = True
                for action in AGOFFGRID.keys():
                    possAActions.append(action)
            else:
                possAActions = []
                for action in COSTS.keys():
                    possAActions.append(action)
        else:
            possAActions = []
            for action in AGOFFGRID.keys():
                possAActions.append(action)

        possBActions = []
        if self.agentingrid(state, "B"):
            agBpos = self.getOldPos(statecopy, "B")
            goalBpos = self.getnewpos(self.agBValue_Matrix)
            if agBpos == goalBpos:
                # bPos_flag=True
                for action in AGOFFGRID.keys():
                    possBActions.append(action)
            else:
                for action in COSTS.keys():
                    possBActions.append(action)
        else:
            for action in AGOFFGRID.keys():
                possBActions.append(action)

        possActions = [
            {"A": aAction, "B": bAction}
            for aAction in possAActions
            for bAction in possBActions
        ]

        # print("Possible Action")
        # print(possActions)
        return possActions

    def getSingleCost(self, agAction):
        # print("len(self.grid)", len(self.grid[0]))

        if (
            agAction == "up"
            or agAction == "down"
            or agAction == "right"
            or agAction == "left"
        ):
            return MOVE_COST

        # elif agAction == "stay":
        #     return STAY_COST

        elif agAction == "agoffgrid":
            return EXIT_COST

    def getCost(self, action, state):

        gridcopy = copy.deepcopy(self.grid)

        aAction = action["A"]

        AActionCost = self.getSingleCost(aAction)

        bAction = action["B"]
        BActionCost = self.getSingleCost(bAction)
        # print((AActionCost, BActionCost))
        # print(min(AActionCost, BActionCost))
        # ActionCost = min(AActionCost,BActionCost)
        Sum_ActionCost = AActionCost + BActionCost
        # print("g(n) is ",ActionCost)
        action_cost_vector = [AActionCost, BActionCost, Sum_ActionCost]
        return action_cost_vector
    
    def doSingleAction(self, agId, agPos, agAction):
        # print("len(self.grid)", len(self.grid[0]))
        if agAction == "up":
            x, y = agPos
            if x == 0:
                return (x, y)
            else:
                newx = x - 1
                newy = y
                return (newx, newy)

        elif agAction == "down":
            x, y = agPos
            if x == (len(self.grid) - 1):
                return (x, y)
            else:
                newx = x + 1
                newy = y
                return (newx, newy)

        elif agAction == "right":
            x, y = agPos
            if y == (len(self.grid[0]) - 1):
                return (x, y)
            else:
                newx = x
                newy = y + 1
                return (newx, newy)

        elif agAction == "left":
            x, y = agPos
            if y == 0:
                return (x, y)
            else:
                newx = x
                newy = y - 1
                return (newx, newy)



        elif agAction == "agoffgrid":
            return OFFGRIDLOC

    def doAction(self, action, state):
        tempnewBPos=tuple()
        specialcase = False
        
        gridcopy = copy.deepcopy(self.grid)
        
        tempgridcopy = copy.deepcopy(self.grid)
        agB_is_present = False
        if self.agentingrid(tempgridcopy, "B"):
            agB_is_present = True
            tempagBpos = self.getOldPos(tempgridcopy, "B")
            tempbAction = action["B"]
            tempnewBPos = self.doSingleAction("B", tempagBpos, tempbAction)
            if tempnewBPos == None:
                agB_is_present = False
            if tempnewBPos:
                if gridcopy[tempnewBPos]=={"T"}:
                    tempnewBPos = tempagBpos

        if self.agentingrid(gridcopy, "A"):
            agApos = self.getOldPos(gridcopy, "A")
            aAction = action["A"]
            newAPos = self.doSingleAction( 
                "A", agApos, aAction
            ) 

            goalApos = self.getnewpos(self.agAValue_Matrix)

            if agB_is_present:
                # print(tempnewBPos)
                if agApos == goalApos:
                    gridcopy[agApos] = {"_"}
                elif gridcopy[newAPos]=={"B"}:
                    if newAPos!=tempnewBPos and tempnewBPos!=agApos:
                        specialcase = True
                        gridcopy[newAPos] = {"A"}
                        if newAPos != agApos:
                            gridcopy[agApos] = {"_"}
                    else: 
                        gridcopy[agApos] = {"A"}
                elif (
                    gridcopy[newAPos] == {"T"}
                ):
                    gridcopy[agApos] = {"A"}
                    gridcopy[newAPos] = {"T"}
                else:
                    if newAPos != agApos:
                        gridcopy[newAPos] = {"A"}
                        gridcopy[agApos] = {"_"}
                    else:
                        gridcopy[newAPos] = {"A"}
            else:
                # print(tempnewBPos)
                if agApos == goalApos:
                    gridcopy[agApos] = {"_"}
                elif tempnewBPos == None:
                    specialcase = True
                    if gridcopy[newAPos] == {"T"}:
                        gridcopy[agApos] = {"A"}
                    else:
                        gridcopy[newAPos] = {"A"}
                        if newAPos != agApos:
                            gridcopy[agApos] = {"_"}
                elif (
                    gridcopy[newAPos] == {"T"}
                    or gridcopy[newAPos] == {"O"}
                ):
                    gridcopy[agApos] = {"A"}
                else:
                    if newAPos != agApos:
                        gridcopy[newAPos] = {"A"}
                        gridcopy[agApos] = {"_"}
                    else:
                        gridcopy[newAPos] = {"A"}

        else:
            pass
        if specialcase:
            agBpos = tempagBpos
            bAction = tempbAction
            newBPos = tempnewBPos
            goalBpos = self.getnewpos(self.agBValue_Matrix)
            if agBpos == goalBpos:
                    if gridcopy[agBpos] == {"A"}:
                        gridcopy[agBpos] = {"A"}
                    else:
                        gridcopy[agBpos] = {"_"}
            elif (
                gridcopy[newBPos] == {"A"}
            ):
                gridcopy[agBpos] = {"B"}
                gridcopy[newBPos] = {"A"}
            elif (
                gridcopy[newBPos] == {"T"}
            ):
                gridcopy[agBpos] = {"B"}
                gridcopy[newBPos] = {"T"}
            else:
                if newBPos != agBpos:
                    gridcopy[newBPos] = {"B"}
                    if gridcopy[agBpos] == {"A"}:
                        gridcopy[agBpos] = {"A"}
                    else:
                        gridcopy[agBpos] = {"_"}
                else:
                    gridcopy[newBPos] = {"B"}
            # print("gridcopy after processing BBBBBBBB special case")
            # print(gridcopy)
        else:
            if self.agentingrid(gridcopy, "B"):
                # print("gridcopy before processing BBBBBBBB")
                # print(gridcopy)
                # bPos = self.getAgPos('B')
                agBpos = self.getOldPos(gridcopy, "B")

                bAction = action["B"]
                newBPos = self.doSingleAction("B", agBpos, bAction)
                goalBpos = self.getnewpos(self.agBValue_Matrix)

                # print("goalBpos", goalBpos)
                # print("agBpos",agBpos)

                if agBpos == goalBpos:
                    if gridcopy[agBpos] == ("A"):
                        gridcopy[agBpos] = {"A"}
                    else:
                        gridcopy[agBpos] = {"_"}
                elif (
                gridcopy[newBPos] == {"A"}
                ):
                    gridcopy[agBpos] = {"B"}
                    gridcopy[newBPos] = {"A"}
                elif (
                    gridcopy[newBPos] == {"T"}
                ):
                    gridcopy[agBpos] = {"B"}
                    gridcopy[newBPos] = {"T"}
                else:
                    if newBPos != agBpos:
                        gridcopy[newBPos] = {"B"}
                        gridcopy[agBpos] = {"_"}
                    else:
                        gridcopy[newBPos] = {"B"}
                # print("gridcopy after processing BBBBBBBB")
                # print(gridcopy)
            else:
                pass


        return AgentGridState(gridcopy, self.agAValue_Matrix, self.agBValue_Matrix)


    def heuristic(self):
        """
        For use with informed search. Returns the cost of reaching
        the goal from this state
        """
        # return 0
        # return self.get_hamming_distance(goal)
        return self.getDistance()

    def getDistance(self):
        """Computes the distance from this state to other"""
        agAdistance = 0
        agBdistance = 0

        gridcopy = copy.deepcopy(self.grid)

        goalApos = self.getnewpos(self.agAValue_Matrix)
        goalBpos = self.getnewpos(self.agBValue_Matrix)

        if self.agentingrid(self, "A"):
            # aPos = self.getAgPos('A')
            aPos = self.getOldPos(gridcopy, "A")
            agA_Curr_Value = self.getAgPosVal(aPos, self.agAValue_Matrix)
            agA_Expected_Value = np.max(self.agAValue_Matrix)
            agAdistance += abs(agA_Curr_Value - agA_Expected_Value)
        else:
            agAdistance = 0

        if self.agentingrid(self, "B"):
            # bPos = self.getAgPos('B')
            bPos = self.getOldPos(gridcopy, "B")
            agB_Curr_Value = self.getAgPosVal(bPos, self.agBValue_Matrix)
            agB_Expected_Value = np.max(self.agBValue_Matrix)
            agBdistance += abs(agB_Curr_Value - agB_Expected_Value)
        else:
            agBdistance = 0

        max_agdistance = max(agAdistance, agBdistance)
        h_score_vector = [agAdistance, agBdistance, max_agdistance]
        # h_score_vector = [agAdistance, agBdistance] #for now just passing one distance
        return h_score_vector

def AgentSearch(initial, goal):
    Agentsearch(initial, goal)

def searchsolution(ag):
    plan = Agentsearch.searchsolution(ag)
    return plan