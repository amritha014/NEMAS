import copy
import random
from itertools import chain

import mdp_gym.utils_gym as ut
import mdptoolbox.util as _util
import numpy as np
from mdp_gym.exceptions import EpisodeDoneError, InvalidActionError


class Env(object):
    """
    Abstract Environment wrapper.
    """
    def __init__(self, seed):
        """
        :param seed: A seed for the random number generator.
        """
        self.set_seed(seed)

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)

class MDP(Env):
    """A Markov Decision Problem.

    Let ``S`` = the number of states, and ``A`` = the number of acions.

    Parameters
    ----------
    transitions : array
        Transition probability matrices. These can be defined in a variety of
        ways. The simplest is a numpy array that has the shape ``(A, S, S)``,
        though there are other possibilities. It can be a tuple or list or
        numpy object array of length ``A``, where each element contains a numpy
        array or matrix that has the shape ``(S, S)``. This "list of matrices"
        form is useful when the transition matrices are sparse as
        ``scipy.sparse.csr_matrix`` matrices can be used. In summary, each
        action's transition matrix must be indexable like ``transitions[a]``
        where ``a`` ∈ {0, 1...A-1}, and ``transitions[a]`` returns an ``S`` ×
        ``S`` array-like object.
    reward : array
        Reward matrices or vectors. Like the transition matrices, these can
        also be defined in a variety of ways. Again the simplest is a numpy
        array that has the shape ``(S, A)``, ``(S,)`` or ``(A, S, S)``. A list
        of lists can be used, where each inner list has length ``S`` and the
        outer list has length ``A``. A list of numpy arrays is possible where
        each inner array can be of the shape ``(S,)``, ``(S, 1)``, ``(1, S)``
        or ``(S, S)``. Also ``scipy.sparse.csr_matrix`` can be used instead of
        numpy arrays. In addition, the outer list can be replaced by any object
        that can be indexed like ``reward[a]`` such as a tuple or numpy object
        array of length ``A``.
    discount : float
        Discount factor. The per time-step discount factor on future rewards.
        Valid values are greater than 0 upto and including 1. If the discount
        factor is 1, then convergence is cannot be assumed and a warning will
        be displayed. Subclasses of ``MDP`` may pass ``None`` in the case where
        the algorithm does not use a discount factor.
    epsilon : float
        Stopping criterion. The maximum change in the value function at each
        iteration is compared against ``epsilon``. Once the change falls below
        this value, then the value function is considered to have converged to
        the optimal value function. Subclasses of ``MDP`` may pass ``None`` in
        the case where the algorithm does not use an epsilon-optimal stopping
        criterion.
    max_iter : int
        Maximum number of iterations. The algorithm will be terminated once
        this many iterations have elapsed. This must be greater than 0 if
        specified. Subclasses of ``MDP`` may pass ``None`` in the case where
        the algorithm does not use a maximum number of iterations.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.

    Attributes
    ----------
    P : array
        Transition probability matrices.
    R : array
        Reward vectors.
    V : tuple
        The optimal value function. Each element is a float corresponding to
        the expected value of being in that state assuming the optimal policy
        is followed.
    discount : float
        The discount rate on future rewards.
    max_iter : int
        The maximum number of iterations.
    policy : tuple
        The optimal policy.
    time : float
        The time used to converge to the optimal policy.
    verbose : boolean
        Whether verbose output should be displayed or not.

    Methods
    -------
    run
        Implemented in child classes as the main algorithm loop. Raises an
        exception if it has not been overridden.
    setSilent
        Turn the verbosity off
    setVerbose
        Turn the verbosity on

    """

    def __init__(self, transitions, reward, discount, terminal_states,size,p0, epsilon, max_iter,seed,
                 skip_check=False):
        # Initialise a MDP based on the input parameters.
        # if the discount is None then the algorithm is assumed to not use it
        # in its computations
        super().__init__(seed)
        self.transitions = transitions
        self.reward = reward
        self.terminal_states=terminal_states
        # if discount is not None:
        #     self.discount = float(discount)
        #     assert 0.0 < self.discount <= 1.0, (
        #         "Discount rate must be in ]0; 1]"
        #     )
        #     if self.discount == 1:
        #         print("WARNING: check conditions of convergence. With no "
        #               "discount, convergence can not be assumed.")

        # # if the max_iter is None then the algorithm is assumed to not use it
        # # in its computations
        # # print(max_iter)
        # # print(epsilon)
        # # print(seed)
        # if max_iter is not None:
        #     self.max_iter = int(max_iter)
        #     assert self.max_iter > 0, (
        #         "The maximum number of iterations must be greater than 0."
        #     )

        # check that epsilon is something sane
        if epsilon is not None:
            self.epsilon = float(epsilon)
            assert self.epsilon > 0, "Epsilon must be greater than 0."

        if not skip_check:
            # We run a check on P and R to make sure they are describing an
            # MDP. If an exception isn't raised then they are assumed to be
            # correct.
            _util.check(transitions, reward)

        # self.S, self.A = _computeDimensions(transitions)
        # # print("TESTING A")
        # # print(transitions)
        P = transitions
        R = reward
        self.state_space = P.shape[1]
        self.action_space = R.shape[1]

        if not skip_check: assert self.state_space == P.shape[2], '3rd Dimension of Transition Matrix is not of size |S|'
        if not skip_check: assert self.action_space == P.shape[0], '2nd Dimension of Transition Matrix is not of size |A|'
        if not skip_check: assert self.state_space == R.shape[0], '1st Dimesnion of Reward Matrix is not of size |S|'

        if not skip_check: assert self.state_space == p0.shape[0], 'Distribution over initial states is not over |S|'
        self.p0 = p0
        self.terminal_states = terminal_states
        self.current_state = None
        self.reset()
        # self.P = P
        # self.R = R

        
        # self.done = False
        # the verbosity is by default turned off
        self.verbose = False
        # Initially the time taken to perform the computations is set to None
        self.time = None
        # set the initial iteration count to zero
        self.iter = 0
        # V should be stored as a vector ie shape of (S,) or (1, S)
        self.V = None
        # policy can also be stored as a vector
        self.policy = None

         
    # def __init__(self, P, R, discount, epsilon, max_iter, skip_check=False):
    #     """
    #     A simple MDP simulator.
    #     :param P: The transition matrix of size |S|x|A|x|S|
    #     :param R: The reward criterion |S|x|A|
    #     :param gamma: the discount factor.
    #     """
    #     super().__init__(seed)
    #     if not skip_check: assert np.allclose(P.sum(axis=2), 1), 'Transition matrix does not seem to be a stochastic matrix ' \
    #                                        '(i.e. the sum over states for each action doesn not equal 1'
    #     self.P = P
    #     self.R = R
    #     self.state_space = P.shape[0]
    #     self.action_space = R.shape[1]
    #     if not skip_check: assert self.state_space == P.shape[2], '3rd Dimension of Transition Matrix is not of size |S|'
    #     if not skip_check: assert self.action_space == P.shape[1], '2nd Dimension of Transition Matrix is not of size |A|'
    #     if not skip_check: assert self.state_space == R.shape[0], '1st Dimesnion of Reward Matrix is not of size |S|'
    #     self.gamma = gamma
    #     if not skip_check: assert self.state_space == p0.shape[0], 'Distribution over initial states is not over |S|'
    #     self.p0 = p0
    #     self.terminal_states = terminal_states
    #     self.current_state = None
    #     self.reset()

    def reset(self):
        integer_representation = np.random.choice(np.arange(self.state_space), p=self.p0)
        self.current_state = ut.convert_int_rep_to_onehot(integer_representation, self.state_space)
        self.done = False
        return self.current_state

    def set_current_state_to(self, state):
        self.current_state = ut.convert_int_rep_to_onehot(state, self.state_space)
        # self.done = False

        return self.current_state
        
    def get_current_state_idx(self):
        # print("PRINT TESTTTT1111111111")
        current_state_idx = ut.convert_onehot_to_int(self.current_state)
        # print(current_state_idx)
        return current_state_idx

    def set_agent_step_to(self,old_agent_step, new_agent_state, action):
        current_state_idx = new_agent_state
        # print(current_state_idx)
        # print(old_agent_step,action)
        reward = self.reward[old_agent_step, action]
        # print("REWARD FROM COMMON.PY",reward)
        if self.current_state.argmax() in self.terminal_states:
            self.done = True        
        else:
            self.done = False
        # print("TUPLE FROM COMMON.PY", current_state_idx, reward, self.done, {'gamma':self.discount})
        return current_state_idx, reward, self.done, {'gamma':self.discount}

    #def set_current_state_to(self, state):
    #    self.current_state = ut.convert_int_rep_to_onehot(state, self.state_space)
    #    # self.done = False
    #    current_state_idx = ut.convert_onehot_to_int(self.current_state)
    #    reward = self.reward[current_state_idx, action]
    #   if self.current_state.argmax() in self.terminal_states:
    #        self.done = True  
        
    #    return self.current_state

        

    def step(self, action):
        """
        :param action: An integer representing the action taken.
        :return:
        """

        # for a in action:
        #     print(a)

        if self.done:
            raise EpisodeDoneError('The episode has terminated. Use .reset() to restart the episode.')

        if action >= self.action_space or not isinstance(action, int):
            raise InvalidActionError('Invalid action {}. It must be an integer between 0 and {}'.format(action, self.action_space-1))

        # we end from this episode onwards.
        # this check is done after entering terminal state
        # because we can only give the reward after leaving
        # a terminal state.
        
        # if self.current_state.argmax() in self.terminal_states:
        #     self.done = True
        
        # get the vector representing the next state probabilities:
        # print("CHECKING CURRENT STATE BEFRE PERFOMONG ACTIONNNNNNNNNNNNNNNNNNNNNNNNNN")
        # print(self.current_state)
        current_state_idx = ut.convert_onehot_to_int(self.current_state)
        next_state_probs = self.transitions[action,current_state_idx]

        # sample the next state

        sampled_next_state = self.rng.choice(np.arange(self.state_space), p=next_state_probs)
        # observe the reward
        reward = self.reward[current_state_idx, action]
        self.current_state = ut.convert_int_rep_to_onehot(sampled_next_state, self.state_space)
        current_state_idx = ut.convert_onehot_to_int(self.current_state)
        if self.current_state.argmax() in self.terminal_states:
            self.done = True        
        return current_state_idx, reward, self.done, {'gamma':self.discount}

