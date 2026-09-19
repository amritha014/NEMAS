VisitedStates = {}
VisitedNode = []

Gopen=set()
Gclosed = set()
OPEN = []
SG=[]
GOALN = set()
COSTS = set()

class Queue:
    """
    A Queue class to be used in combination with state space
    search. The enqueue method adds new elements to the end. The
    dequeue method removes elements from the front.
    """

    def __init__(self):
        self.queue = []

    def __str__(self):
        result = "Queue contains " + str(len(self.queue)) + " items\n"
        for item in self.queue:
            result += str(item) + "\n"
        return result

    def enqueue(self, node):
        self.queue.append(node)

    def dequeue(self):
        if not self.empty():
            return self.queue.pop(0)
        else:
            raise RunTimeError

    def empty(self):
        return len(self.queue) == 0


class Node:
    """
    A Node class to be used in combination with state space search.
    A node contains a state, a parent node, the name of the operator
    used to reach the current state, and the depth of the node in
    the search tree.  The root node should be at depth 0. The method
    repeatedState can be used to determine if the current state
    is the same as the parent's parent's state. Eliminating such
    repeated states improves search efficiency.
    """
    def __init__(self, state, parent, g_score, f_score,operator):
        self.state = state
        self.parent = parent
        self.g_score = g_score
        self.f_score = f_score
        self.operator = operator


    def __str__(self):
        result = "State: \n" + str(self.state)
        result += " \ng_score/path_cost: \n" + str(self.g_score)
        result += " F_score: " + str(self.f_score)
        if self.operator!=None:
            # print(self.operator)
            result += "operator" + str(self.operator)
        if self.parent != None:
            result += " \nParent State: \n" + str(self.parent.state)
        return result

   
    def repeatedState(self):
        global VisitedStates
        # print(VisitedStates)
        if str(self.state) in VisitedStates:
            return 1

        else:
            # print(self)
            VisitedNode.append((str(self.state), str(self.g_score)))
            VisitedStates[str(self.state)] = True
            return 0
            # if self.parent == None: return 0
            # if self.parent.state.equals(self.state): return 1
            # if self.parent.parent == None: return 0
            # if self.parent.parent.state.equals(self.state): return 1
            # return 0

       

class Search:
    """
    A general Search class that can be used for any problem domain.
    Given instances of an initial state and a goal state in the
    problem domain, this class will print the solution or a failure
    message.  The problem domain should be based on the ProblemState
    class.
    """

    def __init__(self, initialState, goalState):
        self.clearVisitedStates()
        self.q = Queue()
        self.q.enqueue(Node(initialState, None, None, 0))
        self.goalState = goalState
        solution = self.execute()
        self.solution=solution
        if solution == None:
            print("Search failed")
        else:
            plan = self.showPath(solution)
            print("\n=======PLAN==========\n",plan)
            self.plan = plan

    def searchsolution(self):
        plan = self.showPath(self.solution)
        return plan
    def clearVisitedStates(self):
        global VisitedStates
        VisitedStates = {}

    def execute(self):
        while not self.q.empty():
            current = self.q.dequeue()
            if self.goalState.equals(current.state):
                return current
            else:
                ActionCost = []
                successors = current.state.applyOperators()
                operators = current.state.operatorNames()
                ActionCost.append(
                    [current.state.getCost(op, current.state) for op in operators]
                )
                for i in range(len(successors)):
                    if not successors[i].illegal():
                        n = Node(
                            successors[i], current, operators[i], current.depth + 1
                        )
                        if n.repeatedState(self.q):
                            del n
                        else:
                            self.q.enqueue(n)
                            # Uncomment the line below to see the queue.
                            # print "Enqueuing state: " + str(n)
        return None

    def showPath(self, GOALN):
        plan = []
        path = self.buildPath(GOALN)
        # print(path)
        # print(" SEARCH PATH:")
        print(" SEARCH PATH:", file=open("Path_output.txt", "a"))
        for key,val in path.items():
            # print("\n PATH ",+ key)
            print("\n PATH ",+ key, file=open("Path_output.txt", "a"))
            for node in val:
                # print(node)
                # print("\n\n",node.operator)
                print("\n\n",node.operator, file=open("Path_output.txt", "a"))
                # print("\n\n",node.state)
                plan.append(node.operator)
        return plan
                
        


    def buildPath(self, GOALN):
        """
        Beginning at the goal node, follow the parent links back
        to the start state.  Create a list of the states traveled
        through during the search from start to finish.
        """
        # print("\nchekc",len(GOALN))
        res = []
        result = {}
        i=0
        # print("GOALN\n", GOALN)
        for node in GOALN:
            res = []
            # print("NODE IN GOALN", node)
            while node!=None:
                # print(node)
                res.insert(0, node)
                node = node.parent
            result[i] = res
            i+=1
            

        # print("result\n",result)
        return result



class ProblemState:
    """
    An interface class for problem domains.
    """

    def illegal(self):
        """
        Tests the state instance for validity.
        Returns true or false.
        """
        abstract()

    def applyOperators(self):
        """
        Returns a list of successors to the current state,
        some of which may be illegal.
        """
        abstract()

    def operatorNames(self):
        """
        Returns a list of operator names in the same order
        as the successors list is generated.
        """
        abstract()

    def equals(self, state):
        """
        Tests whether the state instance equals the given
        state.
        """
        abstract()