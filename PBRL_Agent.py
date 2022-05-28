"""
Implementation of ActorCritic model
"""
import numpy as np

class ActorCritic(object): #actor-critic

        def __init__(self, n_actions, n_states, alpha_theta, alpha_w, gamma):
            self.n_actions = n_actions
            self.n_states = n_states
            self.alpha_theta = alpha_theta
            self.alpha_w= alpha_w
            self.gamma = gamma
            self.theta = np.zeros((n_states, n_actions))
            self.w = np.zeros(n_states)
            self.I = 1
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
