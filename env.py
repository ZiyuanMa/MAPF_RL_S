import numpy as np
import matplotlib as plt

direc = [np.array([0, 1]),np.array([0, -1]),np.array([-1, 0]),np.array([1, 0])]

class History:
    def __init__(self):
        pass


class Environment:
    def __init__(self, env_size, num_agents):
        '''
        self.world:
            0 = empty
            positive = agent with its corresponding id
            -1 = stuff
        '''
        self.num_agents = num_agents
        self.env_size = env_size
        self.world = np.zeros(env_size)
        self.goal = np.zeros(env_size)
        self.agents = []
        self.goals = []
        for i in range(num_agents):

            # agent init position
            x, y = np.random.randint(env_size[0]), np.random.randint(env_size[1])
            self.agents.append(np.array([x, y]))
            self.world[x, y] = i+1

            # agent goal position
            x, y = np.random.randint(env_size[0]), np.random.randint(env_size[1])
            self.goals.append(np.array([x, y]))
            self.goal[x, y] = i+1
        
    def step(self, actions: list):
        '''
        actions:
            list of indices
                0 stay
                1 up
                2 down
                3 left
                4 right

        rewards:
            stay: -0.2
            collusion: -1
            out of bound: -1

        '''

        assert len(actions) == self.num_agents, 'actions number'

        returns = []
        temp_pos = []

        for i, action_idx in enumerate(actions):
            if action_idx == 0:
                returns.append(-0.1)
                continue

            new_pos = self.agents[i] + direc[action_idx-1]

            if new_pos[0] < 0 or new_pos[0] >= self.env_size[0] or new_pos[1] < 0 or new_pos[1] >= self.env_size[1]:
                # out of bound
                returns.append(-1)
                continue

            if self.world[new_pos] == -1:
                returns.append(-1)
                continue

            temp_pos.append(new_pos)


    def observe(self, agent_id):
        '''
        return shape (3, 10, 10)
        first layer: current position
        2nd layer: goal position
        3rd layer: environment
        '''
        assert agent_id >= 1 and agent_id <= self.num_agents, 'agent id out of range'

        obs = np.zeros((3,10,10))

        obs[0,:,:] = self.world==agent_id
        obs[1,self.goals[agent_id-1]] == 1
        obs[0,:,:] = self.world==-1

        return obs
    
    def render(self):

        pass






