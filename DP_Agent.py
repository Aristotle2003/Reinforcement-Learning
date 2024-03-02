"""
COMS W4701 Artificial Intelligence - Programming Homework 3

A dynamic programming agent for a stochastic task environment
"""

import random
import math
import sys


class DP_Agent(object):

    def __init__(self, states, parameters):
        self.gamma = parameters["gamma"]
        self.V0 = parameters["V0"]

        self.states = states
        self.values = {}
        self.policy = {}

        for state in states:
            self.values[state] = parameters["V0"]
            self.policy[state] = None


    def setEpsilon(self, epsilon):
        pass

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        pass


    def choose_action(self, state, valid_actions):
        return self.policy[state]

    def update(self, state, action, reward, successor, valid_actions):
        pass


    def value_iteration(self, valid_actions, transition):
        delta = float('inf')  # Initialize delta to track changes in value
        threshold = 1e-6  # Convergence threshold

        while delta > threshold:
            delta = 0  # Reset delta for this iteration
            for state in self.states:
                # Compute the value for each action and choose the maximum
                max_value = max(transition(state, action)[1] + self.gamma * self.values.get(transition(state, action)[0], 0) for action in valid_actions(state))
                value_change = abs(max_value - self.values[state])  # Calculate the absolute change in value
                self.values[state] = max_value  # Update the value of the state
                delta = max(delta, value_change)  # Update delta



    def policy_extraction(self, valid_actions, transition):
        for state in self.states:
            best_action = None
            best_value = float('-inf')  # Initialize with negative infinity to ensure any real value is higher

            for action in valid_actions(state):
                successor, reward = transition(state, action)
                value = reward + self.gamma * self.values.get(successor, 0)  # Calculate the total value

                if value > best_value:  # Check if this action is better than the current best
                    best_value = value
                    best_action = action

            self.policy[state] = best_action  # Update the policy with the best action for this state

