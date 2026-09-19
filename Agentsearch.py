import random
from itertools import zip_longest

import pygmo as pg
from pq import *
from search import *


class InformedNode(Node):
    """
    Added the goal state as a parameter to the constructor.  Also
    added a new method to be used in conjunction with a priority
    queue.
    """

    def __init__(self, goal, state, parent, g_score, f_score,operator):
        Node.__init__(self, state, parent, g_score, f_score,operator)
        self.goal = goal

    def __str__(self):
        return Node.__str__(self)

    def priority(self):
        """
        Needed to determine where the node should be placed in the
        priority queue.  Depends on the current depth of the node as
        well as the estimate of the distance from the current state to
        the goal state.
        """

        return self.g_score + self.state.heuristic(self.goal)


class Agentsearch(Search):
    """
    A general informed search class that uses a priority queue and
    traverses a search tree containing instances of the InformedNode
    class.  The problem domain should be based on the
    InformedProblemState class.
    """

    def __init__(self, initialState, goalState, pattern):
        self.expansions = 0
        self.clearVisitedStates()
        self.q = PriorityQueue()
        # self.initialState = initialState
        self.goalState = goalState
        self.pattern = pattern
        self.q.enqueue(InformedNode(goalState, initialState, None, [0,0,0], [0,0,0],None))
        initial = InformedNode(goalState, initialState, None, [0,0,0], [0,0,0],None)
        # self.q.enqueue(InformedNode(goalState, initialState, None, [0,0], [0,0],None))
        # initial = InformedNode(goalState, initialState, None, [0,0], [0,0],None)
        self.plan = None
        solution = self.execute(initial)
        if solution == None:
            print("Search failed")
        else:
            self.showPath(solution)
            # print("Expanded", self.expansions, "nodes during search")
    
    def choose_from_open(self,OPEN):
        dominated={}
        # print("\n\nOPEN\n\n",OPEN)
        solutions = set()
        sol=set()
        cand_set = set(OPEN)
        # print(cand_set)
        # print("\n\nLENGTH OF SUCCESORS in OPEN:", len(cand_set))
        if (len(cand_set))==1:
            for ele in cand_set:
                # print("ele in cand_set",ele)
                solutions.add(ele)
                sol = solutions
                    
        else:

            for eachnode in cand_set:
                dominated[eachnode]=False
            for eachnode in cand_set:
                if dominated[eachnode]:
                    continue
                for eachnode1 in cand_set - {eachnode}:
                    if dominated[eachnode1]:
                        continue
                    # print("eachnode1",eachnode1)
                    if pg.pareto_dominance(eachnode[2],eachnode1[2]):
                        dominated[eachnode1] = True
                        if eachnode not in solutions:
                            solutions.add(eachnode)
                        if eachnode1 in solutions:
                            solutions.remove(eachnode1)
                    else:
                        if eachnode not in solutions:
                            solutions.add(eachnode)
                        if eachnode1 not in solutions:
                            solutions.add(eachnode1)
            if solutions:
                # print("solutions",solutions)
                sol.add(min(solutions, key = lambda t: t[2]))
            else: 
                sol.add(min(cand_set, key = lambda t: t[2]))

            
        # print("\n\nsol\n",sol)
        return sol
    
    def compare_dom(self,f_score,g_score):
        if pg.pareto_dominance(f_score,g_score):
            return True
        else:
            return False

         
    def f_score_dominates_COSTS(self,f_score,COSTS):

        for eachcost in COSTS:
            # print("\nEACHCOST IN COSTS:", eachcost)
            if pg.pareto_dominance(f_score,COSTS):
                dominated = False
            if dominated is False:
                return True
                break
        
        return False

    def execute(self,initial):
        GOALN=set()
        # print(GOALN)
        start = initial
        # print("start_node",start)
        OPEN.append((start.state, tuple(start.g_score), tuple(start.f_score),start.parent,start.operator))
        Gopen.add((start.state,tuple(start.g_score),start.parent,start.operator))

        # print("OPEN",OPEN)
        # print("Gopen",Gopen)
        VisitedStatesList=[]
        M=set()
        while OPEN:
            L=[]
            
            #choose non dominated solution from OPEN
            current_set = self.choose_from_open(OPEN)
            # print(current_set)
            # L=(node,parent,tuple(g_score),tuple(f_score))
            for ele in current_set:
                L.append((ele[0],ele[3],ele[1],ele[2],ele[4]))
            # for node in current_set:
            #     print("\n\n NODE CHOSEN FROM OPEN IS:\n", node[0])
            for ele in current_set:
                # remove this node form OPEN set
                OPEN.remove(ele)
                # print("\n\n OPEN REMOVING OVER, lenght of open is:", len(OPEN))
                # remove this node form Gopen
                for x in Gopen:
                    # print("X IS THIS",x)
                    # print("\n\nX[0] is THIS", x[0])
                    if x[0] == ele[0]:
                        Gopen.remove(x)
                        # print("\n\nGOPEN REMOVED", len(Gopen))
                        break 
                # add this node to Gclosed  
                Gclosed.add(ele)
                # print("\n\n ADDED TO GCLOSED, length of gclosed", len(Gclosed))

                #set this node's visited to true
                VisitedStates[ele[0]] = True
                VisitedStatesList.append((ele[0],ele[3]))

            
            for ele in current_set:
                # print("\nELE IN CURRENT_SET IS:", ele)
                current = InformedNode(self.goalState,ele[0],ele[3],ele[1],ele[2],ele[4])
                

            if self.goalState.equals(current.state):
                node = current
                result = []
                while node != None:
                    result.insert(0, node)
                    node = node.parent
                # print(result)
                print(" SEARCH PATH:", file=open("Path_output.txt", "a"))
                for current in result:
                    # if current.depth != 0:
                    print("Operator:", current.operator, file=open("Path_output.txt", "a"))
                    print(current.state, file=open("Path_output.txt", "a"))
                # print("***********************************************************************")
                # print("\n\nCURSTATE",current.state)
                # print("\nGOALSTATE",self.goalState)
                GOALN.add(current)
                M.add(current)
                # GOALN.add(current.state)
                # remove all elements from OPEN set whose f_score is dominated by g_score if the current node
                toremove=[]
                for eachnode in OPEN: 
                    # print("\n",eachnode[0])
                    if self.compare_dom(current.g_score,eachnode[2]):
                        toremove.append(eachnode)

                # print(len(OPEN))
                # print(len(toremove))
                for eachnode in toremove:
                    OPEN.remove(eachnode)
                
                
            else: #since we have only one goal
                actioncost_to_reach_here = []
                
                operators = current.state.possibleActions(current.state)
                # if operators == {'A': 'agoffgrid', 'B': 'agoffgrid'}:
                #     print("h")
                actioncost_to_reach_here.append([current.state.getCost(op, current.state) for op in operators])
                successors = [current.state.doAction(op, current.state) for op in operators]

                # print("\nVisitedStates",VisitedStates)
                # print("\n\nactioncost_to_reach_here:",actioncost_to_reach_here)
                for i in range(len(successors)):
                    # print(OPEN)
                    # for node in OPEN:
                        # print(node[0])
                    if successors[i] in VisitedStates:
                        if VisitedStates[successors[i]] == True:
                            for eachnode in VisitedStatesList:
                                if eachnode[1] == successors[i] and eachnode[0] == current.state:
                                    continue

                    pathcost = current.g_score
                    # print("\n PATH COST", pathcost)
                    # g_score = [ pathcost + eachactioncost for eachactioncost in actioncost_to_reach_here[0][i]]
                    
                    #update g_score and f_score for successor
                    g_score=[]
                    f_score=[]
                    # print(pathcost)
                    # print(actioncost_to_reach_here)
                    for (item1,item2) in zip (list(pathcost),actioncost_to_reach_here[0][i]):
                        g_score.append(item1+item2)
                    # print("\n&&&&&&&&&&&&&&&&& SUCCESORS[i]",successors[i])
                    h_score = successors[i].heuristic()
                    opt_vector = current.state.opt_vector(current.state,self.pattern)
                    # print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{opt_vector}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
                    # print(opt_vector)
                    result_score = [g_score_val + h_score_val for g_score_val, h_score_val in zip(g_score, h_score)]
                    result_score[-1] = max(result_score[:-1])
                    f_score = [result_score - opt_vector for result_score, opt_vector in zip(result_score, opt_vector)]
                    f_score[-1] = max(f_score[:-1])
                    
                    skip = False

                    
                    for eachnode in GOALN:
                        if (pg.pareto_dominance(eachnode.f_score,f_score)):
                            skip = True
                            break
                    
                    if skip:
                        continue

                    if successors[i] in VisitedStates:
                        if VisitedStates[successors[i]]:
                            # print(successors[i])
                            dom = False
                            for node in GOALN:
                                # print("node.f_score",node.f_score)
                                # print("f_score",f_score)
                                if self.compare_dom(node.f_score,f_score):
                                    dom = True
                                    break
                            if not dom:
                                OPEN.append((successors[i], tuple(g_score), tuple(f_score),current,tuple(operators[i].items()))) #just passing social welfare
                                Gopen.add((successors[i], tuple(g_score),current,tuple(operators[i].items())))
                    else: 
                        #remove any g_score in Gopen/Gclosed that's dominated by new g_score
                        Gopen_removelist=[]
                        Gclose_removelist=[]
                        dominated = False

                        for eachnode in Gopen:
                            if pg.pareto_dominance(tuple(g_score),eachnode[1]):
                                Gopen_removelist.append(eachnode)
                                
                        if not dominated:
                            for eachnode in Gclosed:
                                if pg.pareto_dominance(tuple(g_score),eachnode[1]):
                                    Gclose_removelist.append(eachnode)

                        if Gopen_removelist or Gclose_removelist:
                            if Gopen_removelist:
                                for eachnode in Gopen_removelist:
                                    Gopen.remove(eachnode)
                            if Gclose_removelist:
                                for eachnode in Gclose_removelist:
                                    Gclosed.remove(eachnode)
                            
                        OPEN.append((successors[i], tuple(g_score), tuple(f_score),current,tuple(operators[i].items()))) #just passing social welfare
                        Gopen.add((successors[i], tuple(g_score),current,tuple(operators[i].items())))
                                                     
        # return None

        if not OPEN: #
            
            return GOALN 
    

    def showPath(self, GOALN):
        plan = []
        path = self.buildPath(GOALN)
        print(path)
        print(" SEARCH PATH:")
        for key, val in path.items():
            # print("\n PATH ", +key)
            for node in val:
                
                plan.append(node.operator)
        self.plan = plan
        return plan  # Return the plan
    
    def searchsolution(self):
        print(self.plan)
        return self.plan  # Return the plan

    def buildPath(self, GOALN):
        res = []
        result = {}
        i = 0
        # print("GOALN\n", GOALN)
        for node in GOALN:
            res = []
            # print("NODE IN GOALN", node)
            while node is not None:
                # print(node)
                res.insert(0, node)
                node = node.parent
            result[i] = res
            i += 1
        # print("result\n", result)
        return result

 

class InformedProblemState(ProblemState):
    """
    An interface class for problem domains used with informed search.
    """

    def heuristic(self, goal):
        """
        For use with informed search.  Returns the estimated
        cost of reaching the goal from this state.
        """
        abstract()
