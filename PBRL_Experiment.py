"""
Contains all code for the experiments
"""
import numpy as np

from helper import LearningCurvePlot
from PBRL_Agent import ActorCritic
from PBRL_Environment import GridWorld_2D, GridWorld_3D, GridWorld_4D

def run_ActorCritic(Env, Agent, n_timesteps):
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

def run_repetitions(n_repetitions, n_timesteps, alpha_theta, alpha_w, gamma):
    LearningCurve = np.zeros(n_timesteps)
    for rep in range(n_repetitions):
        Env = GridWorld_4D()
        A1 = ActorCritic(Env.n_actions, Env.n_states,alpha_theta, alpha_w,gamma)
        A1.gamma = gamma
        LearningCurve += (1/n_repetitions) * run_ActorCritic(Env, A1, n_timesteps)
    return LearningCurve

def experiment(n_repetitions, n_timesteps):
    #experiment 1 the influence of alpha_w;
    #alpha_w [0.01, 0.05, 0.1, 0.5], alpha_theta = 0.1 and gamma = 0.99
    plot = LearningCurvePlot(title = 'results actor critic algorithm for different values of alpha-w')
    alpha_theta = 0.1
    gamma = 0.99
    for alpha_w in [0.01, 0.05, 0.1, 0.5]:
        LearningCurve = run_repetitions(n_repetitions,n_timesteps, alpha_theta, alpha_w,gamma)
        plot.add_curve(LearningCurve, label='alpha-w = {}'.format(alpha_w))
    plot.save("AC_alpha_w")

    #experiment 1 the influence of alpha_w;
    #alpha_w = 0.1, alpha_theta = [0.01, 0.05, 0.1, 0.5] and gamma = 0.99
    plot = LearningCurvePlot(title = 'results actor critic algorithm for different values of alpha-theta')
    alpha_w = 0.1
    gamma = 0.99
    for alpha_theta in [0.01, 0.05, 0.1, 0.5]:
        LearningCurve = run_repetitions(n_repetitions,n_timesteps, alpha_theta, alpha_w,gamma)
        plot.add_curve(LearningCurve, label='alpha-theta = {}'.format(alpha_theta))
    plot.save("AC_alpha_theta")
    
    #experiment 1 the influence of alpha_w;
    #alpha_w = 0.1, alpha_theta = 0.1 and gamma = [0.8, 0.9, 0.99, 1.0]
    plot = LearningCurvePlot(title = 'results actor critic algorithm for different values of gamma')
    alpha_theta = 0.1
    alpha_w = 0.1
    for gamma in [0.8, 0.9, 0.99, 1.0]:
        LearningCurve = run_repetitions(n_repetitions,n_timesteps, alpha_theta, alpha_w,gamma)
        plot.add_curve(LearningCurve, label='gamma = {}'.format(gamma))
    plot.save("AC_gamma")
    pass

if __name__ == "__main__":
    #experiment parameters
    n_repetitions = 10
    n_timesteps = 1000

    experiment(n_repetitions, n_timesteps)
