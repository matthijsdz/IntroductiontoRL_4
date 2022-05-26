"""
Contains all code for the experiments
"""
from helper import LearningCurvePlot
from PBRL_Agent import ActorCritic, QLearningAgent
from PBRL_Environment import GridWorld_2D, GridWorld_3D, GridWorld_4D
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def run_Qlearning(Env, Agent, n_timesteps):
    state = Env.reset()
    LearningCurve = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        a = Agent.select_action(state)
        r,next_state,done = Env.step(a)
        LearningCurve[t] = r
        Agent.update(state, next_state, a, r)
        state = next_state
    return LearningCurve

def run_ActorCritic(Env, Agent, n_timesteps, n_parameter=27):
    state = Env.reset()
    Agent.I = 1
    LearningCurve = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        a = Agent.select_action(state)
        r,next_state,done = Env.step(a)
        LearningCurve[t] = r
        Agent.update(r,state, next_state,a,done)
        Agent.I = Agent.gamma * Agent.I
        state = next_state
    return LearningCurve

def run_repetitions(n_repetitions, n_timesteps, alpha_theta, alpha_w, gamma, method="AC"):
    LearningCurve = np.zeros(n_timesteps)
    for rep in range(n_repetitions):
        Env = GridWorld_4D()
        if method == "AC":
            A1 = ActorCritic(Env.n_actions, Env.n_states,n_parameters,alpha_theta, alpha_w)
            A1.gamma = gamma
            LearningCurve += (1/n_repetitions) * run_ActorCritic(Env, A1, n_timesteps)
        if method == "QL":
            A2 = QLearningAgent(Env.n_actions, Env.n_states, 0.1)
            A2.gamma = gamma
            LearningCurve += (1/n_repetitions) * run_Qlearning(Env, A1, n_timesteps)
    return LearningCurve

def experiment(n_repetitions, n_timesteps):
    plot = LearningCurvePlot(title = 'results actor critic algorithm for different values of alpha-w')
    alpha_theta = 0.1
    gamma = 0.99
    for alpha_w in [0.01, 0.05, 0.1, 0.5]:
        print("alpha_w")
        LearningCurve = run_repetitions(n_repetitions,n_timesteps, alpha_theta, alpha_w,gamma, method="AC")
        plot.add_curve(LearningCurve, label='alpha-w = {}'.format(alpha_w))
    plot.save("AC_alpha_w_4D")
    plot = LearningCurvePlot(title = 'results actor critic algorithm for different values of alpha-theta')
    alpha_w = 0.1
    gamma = 0.99
    for alpha_theta in [0.01, 0.05, 0.1, 0.5]:
        print("alpha_theta")
        LearningCurve = run_repetitions(n_repetitions,n_timesteps, alpha_theta, alpha_w,gamma, method="AC")
        plot.add_curve(LearningCurve, label='alpha-theta = {}'.format(alpha_theta))
    plot.save("AC_alpha_theta_4D")
    plot = LearningCurvePlot(title = 'results actor critic algorithm for different values of gamma')
    alpha_theta = 0.1
    alpha_w = 0.1
    for gamma in [0.8, 0.9, 0.99, 1.0]:
        print("gamma")
        LearningCurve = run_repetitions(n_repetitions,n_timesteps, alpha_theta, alpha_w,gamma, method="AC")
        plot.add_curve(LearningCurve, label='gamma = {}'.format(gamma))
    plot.save("AC_gamma_4D")
    pass

if __name__ == "__main__":
    #experiment parameters
    n_repetitions = 100
    n_timesteps = 1000
    n_parameters = 27
    alpha_theta = 0.1
    alpha_w = 0.05

    experiment(n_repetitions, n_timesteps)
