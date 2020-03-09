import numpy as np
import matplotlib as plt
import config

action_list = np.array([[0, 0],[0, 1],[0, -1],[-1, 0],[1, 0]], dtype=np.int)


def observe(environment, num_agents, agents_pos, goals_pos):

    obs = np.zeros((4, 3, 10, 10), dtype=np.float32)
    for i in range(num_agents):
        obs[i,0,:,:][tuple(agents_pos[i])] = 1
        obs[i,1,:,:][tuple(goals_pos[i])] = 1
        obs[i,2,:,:] = np.copy(environment)

    return obs

class History:
    def __init__(self, environment, num_agents, agents_pos, goals_pos):
        self.environment = environment
        self.num_agents = num_agents
        self.goals_pos = goals_pos
        self.agents_pos = np.expand_dims(agents_pos, axis=0)
        self.actions = np.array([], dtype=np.int).reshape(0,num_agents)
        self.rewards = np.array([], dtype=np.int).reshape(0,num_agents)
        self.steps = 0

    # @property
    # def num_agents(self):
    #     return self.num_agents

    def __len__(self):

        return self.steps

    def __getitem__(self, index):

        assert index < self.steps, 'step index out of history length'

        return observe(self.environment, self.num_agents, self.agents_pos[index], self.goals_pos), np.copy(self.actions[index]), np.copy(self.rewards[index])

    def push(self, agents_pos, actions, rewards):
        self.agents_pos = np.concatenate((self.agents_pos, np.expand_dims(agents_pos, axis=0)))
        self.actions = np.concatenate((self.actions, np.expand_dims(actions, axis=0)))
        self.rewards = np.concatenate((self.rewards, np.expand_dims(rewards, axis=0)))
        self.steps += 1



class Environment:
    def __init__(self, num_agents=config.num_agents, env_size=config.env_size):
        '''
        self.world:
            0 = empty
            positive = agent with its corresponding id
            -1 = stuff
        '''
        self.num_agents = num_agents
        self.env_size = env_size
        self.world = np.random.choice(2, self.env_size, p=[1-config.obstacle_density, config.obstacle_density]).astype(np.float32)

        self.goals = np.empty((self.num_agents, 2), dtype=np.int)
        for i in range(self.num_agents):
            pos = np.random.randint(0, 10, 2)
            while self.world[tuple(pos)] == 1:
                pos = np.random.randint(0, 10, 2)

            self.goals[i] = pos

        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int)
        for i in range(self.num_agents):
            pos = np.random.randint(0, 10, 2)
            while self.world[tuple(pos)] == 1 or any(np.array_equal(pos, _pos) for _pos in self.agents_pos[:i]):
                pos = np.random.randint(0, 10, 2)

            self.agents_pos[i] = pos
        
        self.steps = 0

        self.history = History(np.copy(self.world), self.num_agents, np.copy(self.agents_pos), np.copy(self.goals))

    def reset(self):

        self.world = np.random.choice(2, self.env_size, p=[1-config.obstacle_density, config.obstacle_density]).astype(np.float32)

        self.goals = np.empty((self.num_agents, 2), dtype=np.int)
        for i in range(self.num_agents):
            pos = np.random.randint(0, 10, 2)
            while self.world[tuple(pos)] == 1:
                pos = np.random.randint(0, 10, 2)

            self.goals[i] = pos

        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int)
        for i in range(self.num_agents):
            pos = np.random.randint(0, 10, 2)
            while self.world[tuple(pos)] == 1 or any(np.array_equal(pos, _pos) for _pos in self.agents_pos[:i]):
                pos = np.random.randint(0, 10, 2)

            self.agents_pos[i] = pos
        
        self.steps = 0

        self.history = History(np.copy(self.world), self.num_agents, np.copy(self.agents_pos), np.copy(self.goals))

    def step(self, actions):
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
        assert all([action_idx<config.action_space and action_idx>=0 for action_idx in actions]), 'action index out of range'

        rewards = np.empty(self.num_agents)
        next_pos = np.copy(self.agents_pos)

        action_direc = np.empty((self.num_agents, 2), dtype=np.int)

        for i, action_idx in enumerate(actions):
            action_direc[i] = np.copy(action_list[action_idx])

        next_pos += action_direc

        agent_list = [i for i in range(self.num_agents)]
        # out of region
        for agent_id in agent_list.copy():
            if np.any(next_pos[agent_id]<np.array([0,0])) or np.any(next_pos[agent_id]>=np.asarray(self.env_size)):
                next_pos[agent_id] = np.copy(self.agents_pos[agent_id])
                rewards[agent_id] = config.collision_reward
                agent_list.remove(agent_id)


        # collide obstacle
        for agent_id in agent_list.copy():
            if self.world[tuple(next_pos[agent_id])] == 1:
                next_pos[agent_id] = np.copy(self.agents_pos[agent_id])
                rewards[agent_id] = config.collision_reward
                agent_list.remove(agent_id)

        
        # collide agent 
        while len(np.unique(next_pos, axis=0)) < self.num_agents:

            recover = []
            pos, count = np.unique(next_pos, axis=0, return_counts=True)
            for p, c in zip(pos, count):
                if c > 1:
                    temp = np.squeeze(np.argwhere(np.all(next_pos==p, axis=1))).tolist()
                    next_pos[temp] = self.agents_pos[temp]
                    rewards[temp] = config.collision_reward

            # agent_list = [agent_id for agent_id in agent_list if agent_id not in recover]
            # next_pos[recover] = self.agents_pos[recover]
            # rewards[recover] = config.collision_reward

        assert len(np.unique(next_pos, axis=0)) == self.num_agents, 'duplicated pos '+str(next_pos)+ ' with ' + str(self.agents_pos) + ' and '+str(recover)

        self.agents_pos = np.copy(next_pos)

        self.steps += 1

        self.history.push(np.copy(self.agents_pos), np.copy(actions), np.copy(rewards))

        # check done
        done = False
        if np.all(self.agents_pos==self.goals):
            rewards = np.ones(self.num_agents)
            done = True
        elif self.steps >= config.max_steps:
            done = True
        
        return done

    def get_history(self):

        return self.history

    def joint_observe(self):
        obs = np.zeros((self.num_agents, 3, 10, 10), dtype=np.float32)
        for i in range(self.num_agents):
            obs[i,0][tuple(self.agents_pos[i])] = 1
            obs[i,1][tuple(self.goals[i])] = 1
            obs[i,2,:,:] = np.copy(self.world)

        return obs

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

    t = np.array([[1,2],[4,3],[1,2]])
    print(np.argwhere(np.all(t==np.array([1,2]), axis=1)))