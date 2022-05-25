"""
policy-based RL agent
"""
import numpy as np
from itertools import product

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        self.S = 0
        self.alpha = 0.1
        self.y = 1
        pass

    def select_action(self, state):
        pi = np.full(self.n_actions, (self.epsilon / (self.n_actions - 1)))
        max_x = np.random.choice(np.where(self.Q[state] == max(self.Q[state]))[0])
        pi[max_x] = 1 - self.epsilon
        a = np.random.choice(range(self.n_actions),p=pi)
        return a

    def update(self, S, S_prime, a, reward):
        self.Q[S][a] = self.Q[S][a] + self.alpha * (reward + self.y*max(self.Q[S_prime])-self.Q[S][a])
        pass



class ActorCritic(object): #actor-critic

        def __init__(self, n_actions, n_states, n_parameters, alpha_theta, alpha_w):
            self.n_actions = n_actions
            self.n_states = n_states
            self.alpha_theta = alpha_theta
            self.alpha_w= alpha_w
            self.gamma = 1.0
            self.n_parameters = n_parameters
            self.theta = np.zeros((n_states, n_actions))
            self.w = np.zeros(n_states)
            self.I = 0
            pass

        def pi(self,state):
            h = self.theta[state]
            pi = np.exp(h) / np.sum(np.exp(h))
            return pi

        def select_action(self, state):
            pi = self.pi(state)
            a = np.random.choice(range(self.n_actions),p=pi)
            return a

        def update(self, r, state, next_state, action, done):
            if done == True:
                target = r
            else:
                target = r + self.gamma * self.w[next_state]
            delta = target - self.w[state]
            self.w[state] += self.alpha_w * (target - self.w[state])
            x_s = np.zeros(self.n_actions)
            x_s[action] = 1
            prob = self.pi(state)
            self.theta[state] += self.alpha_theta * delta * self.I * (x_s - prob)
            pass

class ActorCritic_ET(object): #actor-critic

    def __init__(self, n_actions, n_states, n_parameters, alpha_theta, alpha_w):
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha_theta = alpha_theta
        self.alpha_w= alpha_w
        self.gamma = 1.0
        self.n_parameters = n_parameters
        self.theta = np.zeros((n_states, n_actions))
        self.w = np.zeros(n_states)
        self.I = 0
        self.lambda_w = 0
        self.lambda_theta = 0
        self.z_w = 0
        self.z_theta = 0
        pass

    def pi(self,state):
        h = self.theta[state]
        pi = np.exp(h) / np.sum(np.exp(h))
        return pi

    def select_action(self, state):
        pi = self.pi(state)
        a = np.random.choice(range(self.n_actions),p=pi)
        return a

    def update(self, r, state, next_state, action, done):
        if done == True:
            target = r
        else:
            target = r + self.gamma * self.w[next_state]
        delta = target - self.w[state]
        self.z_w = self.gamma*self.lambda_w*self.z_w
        self.z_theta = self.gamma*self.lambda_theta*self.z_theta + self.I * (x_s - prob)
        self.w[state] += self.alpha_w * delta
        x_s = np.zeros(self.n_actions)
        x_s[action] = 1
        prob = self.pi(state)
        self.theta[state] += self.alpha_theta * delta * self.I * (x_s - prob)
        pass
