import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from numba import jit
import config
import random

action_list = np.array([[0, 0],[-1, 0],[1, 0],[0, -1],[0, 1]], dtype=np.int8)


def observe(environment, num_agents, agents_pos, goals_pos):

    obs = np.zeros((num_agents, 3, *config.env_size), dtype=np.float32)
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
        self.rewards = np.array([], dtype=np.float32).reshape(0,num_agents)
        self.steps = 0

    # @property
    # def num_agents(self):
    #     return self.num_agents

    def __len__(self):

        return self.steps

    def __getitem__(self, index):

        assert index < self.steps, 'step index out of history length'
        # print(self.rewards[index])
        return observe(self.environment, self.num_agents, self.agents_pos[index], self.goals_pos), np.copy(self.actions[index]), np.copy(self.rewards[index])

    def push(self, agents_pos, actions, rewards):
        self.agents_pos = np.concatenate((self.agents_pos, np.expand_dims(agents_pos, axis=0)))
        self.actions = np.concatenate((self.actions, np.expand_dims(actions, axis=0)))
        self.rewards = np.concatenate((self.rewards, np.expand_dims(rewards, axis=0)))
        self.steps += 1

    def done(self):
        if np.all(self.agents_pos[-1,:,:] == self.goals_pos):
            return True
        else:
            return False


def map_partition(map, agent_pos):
    open_list = []
    open_list.append(agent_pos)
    close_list = []
    while len(open_list) > 0:
        pos = open_list.pop()

        top = (pos[0]-1, pos[1])
        if top[0] >= 0 and map[top]==0 and top not in open_list and top not in close_list:
            open_list.append(top)
        
        down = (pos[0]+1, pos[1])
        if down[0] <= 7 and map[down]==0 and down not in open_list and down not in close_list:
            open_list.append(down)
        
        left = (pos[0], pos[1]-1)
        if left[1] >= 0 and map[left]==0 and left not in open_list and left not in close_list:
            open_list.append(left)
        
        right = (pos[0], pos[1]+1)
        if right[1] <= 7 and map[right]==0 and right not in open_list and right not in close_list:
            open_list.append(right)

        close_list.append(pos)
    return close_list
    


class Environment:
    def __init__(self, num_agents, env_size=config.env_size):
        '''
        self.map:
            0 = empty
            1 = obstacle
        '''
        self.num_agents = num_agents
        self.env_size = env_size
        self.map = np.random.choice(2, self.env_size, p=[1-config.obstacle_density, config.obstacle_density]).astype(np.float32)

        empty_pos = np.argwhere(self.map==0).astype(np.int8)
        self.goals = empty_pos[np.random.choice(empty_pos.shape[0], self.num_agents, replace=False)]


        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int8)
        for i in range(self.num_agents):

            pos_list = map_partition(self.map, tuple(self.goals[i]))
            pos = np.asarray(random.choice(pos_list))
            while np.any(np.all(self.agents_pos[:i]==pos, axis=1)):
                pos = np.asarray(random.choice(pos_list))

            self.agents_pos[i] = pos
        
        self.steps = 0

        self.history = History(np.copy(self.map), self.num_agents, np.copy(self.agents_pos), np.copy(self.goals))

    def reset(self):

        self.map = np.random.choice(2, self.env_size, p=[1-config.obstacle_density, config.obstacle_density]).astype(np.float32)

        empty_pos = np.argwhere(self.map==0)
        self.goals = empty_pos[np.random.choice(empty_pos.shape[0], self.num_agents, replace=False)]


        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int8)
        for i in range(self.num_agents):

            pos_list = map_partition(self.map, tuple(self.goals[i]))
            pos = np.asarray(random.choice(pos_list))
            while np.any(np.all(self.agents_pos[:i]==pos, axis=1)):
                pos = np.asarray(random.choice(pos_list))

            self.agents_pos[i] = pos
        
        self.steps = 0

        self.history = History(np.copy(self.map), self.num_agents, np.copy(self.agents_pos), np.copy(self.goals))

    def load(self, world, num_agents, agents_pos, goals_pos):
        self.num_agents = num_agents
        self.env_size = (8, 8)
        self.map = np.array(world)
        print(self.map)
        self.goals = np.array(goals_pos, dtype=np.int8)


        self.agents_pos = np.array(agents_pos, dtype=np.int8)

        
        self.steps = 0

        self.history = History(np.copy(self.map), self.num_agents, np.copy(self.agents_pos), np.copy(self.goals))

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

        # assert len(actions) == self.num_agents, 'actions number'
        # assert all([action_idx<config.action_space and action_idx>=0 for action_idx in actions]), 'action index out of range'

        if np.unique(self.agents_pos, axis=0).shape[0] < self.num_agents:
            print(self.steps)
            print(self.map)
            print(self.history.agents_pos[-2])
            print(self.history.actions[-1])
            print(self.agents_pos)
            raise RuntimeError('unique')

        check_id = [i for i in range(self.num_agents)]

        rewards = np.empty(self.num_agents, dtype=np.float32)
        for agent_id in check_id.copy():
            if actions[agent_id] == 0:
                check_id.remove(agent_id)
                if np.array_equal(self.agents_pos[agent_id], self.goals[agent_id]):
                    rewards[agent_id] = config.stay_on_goal_reward
                else:
                    rewards[agent_id] = config.stay_off_goal_reward
            else:
                rewards[agent_id] = config.move_reward


        next_pos = np.copy(self.agents_pos)

        for agent_id in check_id:
            next_pos[agent_id] += np.copy(action_list[actions[agent_id]])




        for agent_id in check_id.copy():

            # move

            if np.any(next_pos[agent_id]<np.array([0,0])) or np.any(next_pos[agent_id]>=np.asarray(self.env_size)): 
                # agent out of bound
                rewards[agent_id] = config.collision_reward
                next_pos[agent_id] = np.copy(self.agents_pos[agent_id])
                check_id.remove(agent_id)

            elif self.map[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                rewards[agent_id] = config.collision_reward
                next_pos[agent_id] = np.copy(self.agents_pos[agent_id])
                check_id.remove(agent_id)

            
        

        flag = False
        while not flag:
            
            flag = True
            for agent_id in check_id.copy():
                
                if len(*np.where(np.all(next_pos==next_pos[agent_id], axis=1))) > 1:
                    # collide agent

                    collide_agent_id = np.where(np.all(next_pos==next_pos[agent_id], axis=1))[0].tolist()
                    collide_agent_id = [ id for id in collide_agent_id if id in check_id]
                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    rewards[collide_agent_id] = config.collision_reward

                    for id in collide_agent_id:
                        check_id.remove(id)

                    flag = False
                    break

                elif np.any(np.all(next_pos[agent_id]==self.agents_pos, axis=1)):
                    # agent swap

                    target_agent_id = np.where(np.all(next_pos[agent_id]==self.agents_pos, axis=1))
                    assert len(target_agent_id) == 1, 'target > 1'

                    if np.array_equal(next_pos[target_agent_id], self.agents_pos[agent_id]):
                        assert target_agent_id in check_id, 'not in check'
                        next_pos[agent_id] = self.agents_pos[agent_id]
                        rewards[agent_id] = config.collision_reward
                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        rewards[target_agent_id] = config.collision_reward
                        check_id.remove(agent_id)
                        check_id.remove(target_agent_id)
                        flag = False
                        break


        self.agents_pos = np.copy(next_pos)

        self.steps += 1

        # check done
        done = False
        if np.all(self.agents_pos==self.goals):
            rewards = np.ones(self.num_agents) * config.finish_reward
            done = True
        elif self.steps >= config.max_steps:
            done = True

        self.history.push(np.copy(self.agents_pos), np.copy(actions), np.copy(rewards))
        
        return done

    def get_history(self):

        return self.history

    def joint_observe(self):
        obs = np.zeros((self.num_agents, 3, 8, 8), dtype=np.float32)
        for i in range(self.num_agents):
            obs[i,0][tuple(self.agents_pos[i])] = 1
            obs[i,1][tuple(self.goals[i])] = 1
            obs[i,2,:,:] = np.copy(self.map)

        return obs

    def observe(self, agent_id):
        '''
        return shape (3, 8, 8)
        first layer: current position
        2nd layer: goal position
        3rd layer: environment
        '''
        assert agent_id >= 0 and agent_id < self.num_agents, 'agent id out of range'

        obs = np.zeros((3,8,8))

        obs[0,:,:][tuple(self.agents_pos[agent_id])] = 1
        obs[1,:,:][tuple(self.goals[agent_id])] = 1
        obs[2,:,:] = np.copy(self.map)

        return obs
    
    def render(self):
        map = np.copy(self.map)
        for agent_id in range(self.num_agents):
            if np.array_equal(self.agents_pos[agent_id], self.goals[agent_id]):
                map[tuple(self.agents_pos[agent_id])] = 4
            else:
                map[tuple(self.agents_pos[agent_id])] = 2
                map[tuple(self.goals[agent_id])] = 3

        cmap = colors.ListedColormap(['white','grey','lime','purple','gold'])
        plt.imshow(map, cmap=cmap)
        plt.xlabel(str(self.steps))
        plt.ion()
        plt.show()
        plt.pause(1)



