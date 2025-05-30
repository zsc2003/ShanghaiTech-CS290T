import logging
import math
from typing import Tuple, List

from tqdm import tqdm
import numpy as np

from .Game import Game

EPS = 1e-8

log = logging.getLogger(__name__)

class PureMCTS():
    def __init__(self, game: Game, args):
        self.game = game
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited

        self.Ps = {}  # For pure MCTS, Ps is the same with Vs

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
    
    def load_tree(self, file):
        try:
            self.Qsa, self.Nsa,  self.Ns, self.Ps, self.Es, self.Vs = np.load(file, allow_pickle=True)
        except Exception as e:
            log.error("Error loading tree: %s", e)
            return
    
    def dump_tree(self, file):
        np.save(file, (self.Qsa, self.Nsa, self.Ns, self.Ps, self.Es, self.Vs))
        
    def getActionProb(self, canonicalBoard, temp=1):
        """
        Calculate the probability of each action based on the number of times each action has been selected.

        Args:
            canonicalBoard (np.ndarray): The current state of the game board.
            temp (float, optional): The temperature parameter for the softmax function. A lower temperature results in more deterministic result, 0 means no sample.

        Returns:
            List: A list of probabilities for each action.
        """

        # doing rollout for numMCTSSims times
        """Your code here"""

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            # return the probs with only one of the best actions (break ties randomly)
            """Your code here"""

        # return the probs with temperature
        # the policy vector where the probability of the ith action is proportional to Nsa[(s,a)]**(1/temp)
        """Your code here"""
    
    def select(self, canonicalBoard) -> Tuple[List[Tuple[np.ndarray, int]], np.ndarray]:
        """
        Selects an unexpanded node on the given canonical board.

        Args:
            canonicalBoard (np.ndarray): The current canonical board state.

        Returns:
            Tuple[List[Tuple[np.ndarray, int]], np.ndarray]: A tuple containing the path of moves made and the final board state.
        """
        # The `select`` here totally covers the phase of selection and expansion
        # for ease of implementation
        path = []
        s = self.game.stringRepresentation(canonicalBoard)
        while True:
            # if node is expanded or terminal return the path and leaf node
            """Your code here"""
            
            # select node by ucb selection to generate a path
            # NOTICE: for the board is always a canonicalBoard, so the current player is always 1
            # use self.game.getNextState(canonicalBoard, 1, action)
            """Your code here"""
            s = self.game.stringRepresentation(canonicalBoard)
    
    def expand(self, canonicalBoard, s):
        """
        Expand the search tree by adding the valid moves for the current state.

        Args:
            canonicalBoard (numpy.ndarray): The current state of the board.
            s (str): The state identifier.

        Returns:
            None
        """
        if s not in self.Ns:
            valids = self.game.getValidMoves(canonicalBoard, 1)
            if len(valids) == 0:
                return
            self.Vs[s] = valids
            self.Ns[s] = 0

    def simulate(self, canonicalBoard, s):
        """
        Simulate the game from the given state until the terminal state is reached.

        Args:
            canonicalBoard (numpy.ndarray): The current state of the game board.
            s (str): The string representation of the current state.

        Returns:
            int: The reward obtained from the simulation.
        """
        invert_reward = True
        while True:
            if s not in self.Es:
                reward = self.game.getGameEnded(canonicalBoard, 1)
                self.Es[s] = reward

            # if the game has ended
            # return the reward
            # NOTICE: beware which one player the reward is for, which can be determined by invert_reward
            """Your code here"""

            if s in self.Vs:
                valids = self.Vs[s]
            else:
                valids = self.game.getValidMoves(canonicalBoard, 1)
                self.Vs = valids
            # get random action from valid moves
            # get the next board and update 's'
            # NOTICE: for the board is always a canonicalBoard, so the current player is always 1
            # use self.game.getNextState(canonicalBoard, 1, action)
            """Your code here"""
            
    def backup(self, path, reward):
        """
        Perform the backup operation for the given path and reward.

        Args:
            path (list): A list of tuples, where each tuple contains a canonical board state and the corresponding action taken.
            reward (float): The reward obtained after taking the action.

        """

        # This method iterates over the path in reverse order, updating Ns, Nsa, and Qsa.
        # NOTICE: the reward is different for different player, so we need to invert it every time
        """Your code here"""

    def search(self, canonicalBoard):
        """
        Perform a search on the given canonical board.
        Doing select, expand, simulate and backup in sequence.

        Args:
            canonicalBoard (object): The current state of the game board.

        Returns:
            None: This method does not return a value.
        """
        # do selection, expansion and simulation
        """Your code here"""

        # expand the node for the root
        if len(path) == 0:
            return

        # do backup
        """Your code here"""
        
        
    def ucb_select(self, s: str, validMoves: np.ndarray) -> int:
        """
        Selects the action with the highest Upper Confidence Bound (UCB) for a given state.

        Args:
            s (str): The string representation of the current state.
            validMoves (np.ndarray): A binary array where each index represents whether 
                                     the corresponding action is valid (1) or not (0).

        Returns:
            int: The index of the action with the highest UCB.
        """
        cur_best = -float('inf')
        best_act = -1
        
        # score = Qsa + cpuct * sqrt(Ns / Nsa)
        # NOTICE: we always select the `first` action that has not been visited
        """Your code here"""

        return best_act


class MCTS(PureMCTS):
    """
    This class handles the MCTS tree for AlphaZero.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s


    def simulate(self, canonicalBoard, s):
        """
        Simulate the game from the given state until the terminal state is reached.

        Args:
            canonicalBoard (numpy.ndarray): The current state of the game board.
            s (str): The string representation of the current state.

        Returns:
            int: The reward obtained from the simulation.
        """
        
        # doing simulation like pure mcts
        # NOTICE: use nnet to get policy and reward
        # NOTICE: store the policy into Ps
        # NOTICE: there is no need to simulate unitl the game ends
        """Your code here"""

    def ucb_select(self, s: str, validMoves: np.ndarray) -> int:
        # ucb select formula: u = value + cpuct * P * sqrt(N) / (1 + Nsa)
        """Your code here"""

