"""
Environment for policy-based machine learning agent
"""

import numpy as np
import pandas as pd

class GridWorld_2D():

    def __init__(self):
        self.width = 5
        self.height = 5
        self.shape = (self.height, self.width)
        self.n_states = self.height * self.width
        self.n_actions = 4
        self.goal = 24
        self.start_location = 0
        self.location = self.start_location

    def get_actions(self):
        return [0,1,2,3]

    def state_to_location(self, state, small=False):
        return np.unravel_index(state, self.shape)

    def location_to_state(self,location):
        return np.ravel_multi_index(location, self.shape)

    def reset(self):
        self.location = self.start_location
        return self.location

    def step(self, action):
        x,y = self.state_to_location(self.location)
        if action == 0:
            if y>0:
                y -= 1
        elif action == 1:
            if y <self.height-1:
                y += 1
        elif action == 2:
            if x>0:
                x -= 1
        elif action == 3:
            if x<self.width-1:
                x += 1
        new_location = self.location_to_state([x,y])
        self.location = new_location
        if new_location == self.goal:
            return 200, self.location, True
        return -1, self.location, False

class GridWorld_3D():

    def __init__(self):
        self.width = 5
        self.height = 5
        self.depth = 5
        self.shape = (self.height, self.width, self.depth)
        self.n_states = self.height * self.width * self.depth
        self.n_actions = 6
        self.goal = 124
        self.start_location = 0
        self.location = self.start_location

    def state_to_location(self, state):
        return np.unravel_index(state, self.shape)

    def location_to_state(self,location):
        return np.ravel_multi_index(location, self.shape)

    def get_actions():
        return [0,1,2,3,4,5] #up/down/left/right/forward/backward

    def reset(self):
        self.location = self.start_location
        return self.location

    def step(self, action):
        x,y,z = self.state_to_location(self.location)
        if action == 0:
            if y>0:
                y -= 1
        elif action == 1:
            if y <self.height-1:
                y += 1
        elif action == 2:
            if x>0:
                x -= 1
        elif action == 3:
            if x<self.width-1:
                x += 1
        elif action == 4:
            if z>0:
                z -= 1
        elif action == 5:
            if z<self.depth-1:
                z += 1
        new_location = self.location_to_state([x,y,z])
        self.location = new_location
        if new_location == self.goal:
            return 200, self.location, True
        return -1, self.location, False

class GridWorld_4D():

    def __init__(self):
        self.width = 5
        self.height = 5
        self.depth = 5
        self.D4 = 5
        self.shape = (self.height, self.width, self.depth, self.D4)
        self.n_states = self.height * self.width * self.depth * self.D4
        self.n_actions = 8
        self.goal = 624
        self.start_location = 0
        self.location = self.start_location

    def state_to_location(self, state):
        return np.unravel_index(state, self.shape)

    def location_to_state(self,location):
        return np.ravel_multi_index(location, self.shape)

    def get_actions():
        return [0,1,2,3,4,5,6,7] #up/down/left/right/forward/backward

    def reset(self):
        self.location = self.start_location
        return self.location

    def step(self, action):
        x,y,z,d4 = self.state_to_location(self.location)
        if action == 0:
            if y>0:
                y -= 1
        elif action == 1:
            if y <self.height-1:
                y += 1
        elif action == 2:
            if x>0:
                x -= 1
        elif action == 3:
            if x<self.width-1:
                x += 1
        elif action == 4:
            if z>0:
                z -= 1
        elif action == 5:
            if z<self.depth-1:
                z += 1
        elif action == 6:
            if d4>0:
                d4 -= 1
        elif action == 7:
            if d4<self.D4-1:
                d4 += 1
        new_location = self.location_to_state([x,y,z,d4])
        self.location = new_location
        if new_location == self.goal:
            return 200, self.location, True
        return -1, self.location, False

if __name__ == "__main__":
    A = GridWorld_1()
