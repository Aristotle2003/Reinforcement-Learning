"""
COMS W4701 Artificial Intelligence - Programming Homework 3

A Q-learning agent for a stochastic task environment
"""

import random
import math
import sys


class RL_Agent(object):

    def __init__(self, states, valid_actions, parameters):
        self.alpha = parameters["alpha"]
        self.epsilon = parameters["epsilon"]
        self.gamma = parameters["gamma"]
        self.Q0 = parameters["Q0"]

        self.states = states
        self.Qvalues = {}
        for state in states:
            for action in valid_actions(state):
                self.Qvalues[(state, action)] = parameters["Q0"]


    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        self.alpha = alpha


    def choose_action(self, state, valid_actions):
    # Exploration vs Exploitation
        if random.random() < self.epsilon:
        # Exploration: Random action
            return random.choice(valid_actions)
        else:
        # Exploitation: Choose the best action based on Q-values
            best_action = None
            max_value = float('-inf')
            for action in valid_actions:
                q_value = self.Qvalues.get((state, action), self.Q0)
                if q_value > max_value:
                    max_value = q_value
                    best_action = action
            return best_action



    def update(self, state, action, reward, successor, valid_actions):
    # Compute Q(s', a') for all a' if successor is not None
        if successor:
            max_future = max([self.Qvalues.get((successor, a), self.Q0) for a in valid_actions])
        else:
            max_future = 0

        # Current Q-value
        current_q = self.Qvalues.get((state, action), self.Q0)

        # Q-learning update formula
        self.Qvalues[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_future - current_q)

