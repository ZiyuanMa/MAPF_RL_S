import numpy as np
import matplotlib as plt
import config

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
        self.world = np.random.choice(2, env_size, p=[0.85, 0.15])

        self.goals = np.empty((num_agents, 2))
        for i in range(num_agents):
            pos = np.random.randint(0, 10, 2)
            while self.world[tuple(pos)] == 1:
                pos = np.random.randint(0, 10, 2)

            self.goals[i] = pos

        self.agents = np.empty((num_agents, 2))
        for i in range(num_agents):
            pos = np.random.randint(0, 10, 2)
            while self.world[tuple(pos)] == 1 or any(np.array_equal(pos, _pos) for _pos in self.agents[:i]):
                pos = np.random.randint(0, 10, 2)

            self.agents[i] = pos


        
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
        assert agent_id >= 0 and agent_id < self.num_agents, 'agent id out of range'

        obs = np.zeros((3,10,10))

        obs[0,:,:][tuple(self.agents[agent_id])] = 1
        obs[1,:,:][tuple(self.goals[agent_id])] = 1
        obs[2,:,:] = np.copy(self.world)

        return obs
    
    def render(self):

        pass






if __name__ == '__main__':
    Environment(config.env_size, config.num_agents)