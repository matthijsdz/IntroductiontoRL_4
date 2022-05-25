"""
Environment for policy-based machine learning agent
"""

import numpy as np
import pandas as pd

class GridWorld_2D():

    def __init__(self):
        #Get-GridWorld-From-File------------
        with open("gridworld.txt") as file:
            x = file.readlines()
        i = 0
        self.goal = 15
        self.walls = []
        self.start_location = 0
        for row in x:
            for char in row.strip():
                if char == "x":
                    self.walls.append(i)
                if char == "G":
                    self.goal = i
                if char == "S":
                    self.start_location = i
                i += 1
        #------------------------------------
        self.width = len(row)
        self.height = len(x)
        self.shape = (self.height, self.width)
        self.location = self.start_location
        self.n_states = self.height * self.width
        self.n_actions = 4

    def Vectorize(self,poly,n_parameters,state,a=None):
        x,y = np.unravel_index(state, self.shape)
        x = x/(self.width)
        y = y/(self.height)
        input = [x,y]
        if a != None:
            input = [state/self.n_states,a/4]
        x = poly.transform([input])[0]
        return x[:n_parameters]

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
        if new_location not in self.walls:
            self.location = new_location
        if new_location == self.goal:
            return 200, self.location, True
        return -1, self.location, False

class GridWorld_3D():

    def __init__(self):
        self.width = 10
        self.height = 10
        self.depth = 10
        self.shape = (self.height, self.width, self.depth)
        self.n_states = self.height * self.width * self.depth
        self.goal = 99
        self.location = 0

    def state_to_location(self, state):
        return np.unravel_index(state, self.shape)

    def location_to_state(self,location):
        return np.ravel_multi_index(location, self.shape)

    def possible_action():
        return [0,1,2,3,4,5] #up/down/left/right/forward/backward

    def reset(self):
        self.location = 0

    def step(self, action):
        x,y,z = self.state_to_location(self.location)
        if action == 0:
            if y>0:
                y -= 1
        elif action == 1:
            if y <self.height-1:
                self.y += 1
        elif action == 2:
            if x>0:
                x -= 1
        elif action == 3:
            if x<self.width-1:
                x += 1
        elif action == 4:
            if x>0:
                z -= 1
        elif action == 5:
            if x<self.depth-1:
                z += 1
        self.location = self.location_to_state([x,y,z])
        if self.location == self.goal:
            return 200
        return -1

if __name__ == "__main__":
    A = GridWorld_1()
