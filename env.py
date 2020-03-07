import numpy as np
import matplotlib as plt
import config

action_direc = np.array([[0, 0],[0, 1],[0, -1],[-1, 0],[1, 0]])

class History:
    def __init__(self):
        pass



class Environment:
    def __init__(self, env_size=config.env_size, num_agents=config.num_agents):
        '''
        self.world:
            0 = empty
            positive = agent with its corresponding id
            -1 = stuff
        '''
        self.num_agents = num_agents
        self.env_size = env_size
        self.world = np.random.choice(2, env_size, p=[1-config.obstacle_density, config.obstacle_density])

        self.goals = np.empty((num_agents, 2))
        for i in range(num_agents):
            pos = np.random.randint(0, 10, 2)
            while self.world[tuple(pos)] == 1:
                pos = np.random.randint(0, 10, 2)

            self.goals[i] = pos

        self.agents_pos = np.empty((num_agents, 2))
        for i in range(num_agents):
            pos = np.random.randint(0, 10, 2)
            while self.world[tuple(pos)] == 1 or any(np.array_equal(pos, _pos) for _pos in self.agents_pos[:i]):
                pos = np.random.randint(0, 10, 2)

            self.agents_pos[i] = pos
        
        self.num_steps = 0


        
    def step(self, actions: list):
        '''
        actions:
            list of indices
                0 stay
                1 up
                2 down
                3 left
                4 right
        '''

        assert len(actions) == self.num_agents, 'actions number'

        rewards = [None for _ in range(self.num_agents)]
        next_pos = np.empty((self.num_agents, 2))

        for i, action_idx in enumerate(actions):
            next_pos[i] = self.agents_pos[i] + action_direc[action_idx]


        agent_list = [i for i in range(self.num_agents)]
        # out of region
        for agent_id in agent_list.copy():
            if np.any(next_pos[agent_id]<np.array([0,0])) or np.any(next_pos[agent_id]>=np.asarray(self.env_size)):
                next_pos[agent_id] = np.copy(self.world[agent_id])
                rewards[agent_id] = config.collision_reward
                agent_list.remove(agent_id)


        # collide obstacle
        for agent_id in agent_list.copy():
            if self.world[tuple(next_pos[agent_id])] == 1:
                next_pos[agent_id] = np.copy(self.world[agent_id])
                rewards[agent_id] = config.collision_reward
                agent_list.remove(agent_id)

        
        # collide agent 
        # for agent_id in agent_list.copy():
        #     if self.world[tuple(next_pos[agent_id])] == 1:
        #         agent_pos = np.copy(self.world[agent_id])
        #         rewards[agent_id] = config.collision_reward
        #     else:
        #         agent_list.append(agent_id)





        


    def observe(self, agent_id):
        '''
        return shape (3, 10, 10)
        first layer: current position
        2nd layer: goal position
        3rd layer: environment
        '''
        assert agent_id >= 0 and agent_id < self.num_agents, 'agent id out of range'

        obs = np.zeros((3,10,10))

        obs[0,:,:][tuple(self.agents_pos[agent_id])] = 1
        obs[1,:,:][tuple(self.goals[agent_id])] = 1
        obs[2,:,:] = np.copy(self.world)

        return obs
    
    def render(self):

        pass






if __name__ == '__main__':
    # Environment(config.env_size, config.num_agents)
    a = np.array([[1,3],[3,4]])
    b = np.array([1,2])
    print(a==b)