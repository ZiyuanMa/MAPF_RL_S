import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from numba import jit
import config
import random

from typing import List

action_list = np.array([[0, 0],[-1, 0],[1, 0],[0, -1],[0, 1]], dtype=np.int8)


def observe(environment, num_agents, agents_pos, goals_pos):

    obs = np.zeros((num_agents, 3, *config.map_size), dtype=np.float32)
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


def map_partition(map):

    empty_pos = np.argwhere(map==0).astype(np.int8).tolist()

    empty_pos = [ tuple(pos) for pos in empty_pos]

    if not empty_pos:
        raise RuntimeError('no empty pos')

    partition_list = list()
    while empty_pos:

        start_pos = empty_pos.pop()

        open_list = list()
        open_list.append(start_pos)
        close_list = list()

        while open_list:
            pos = open_list.pop()

            up = (pos[0]-1, pos[1])
            if up[0] >= 0 and map[up]==0 and up in empty_pos:
                empty_pos.remove(up)
                open_list.append(up)
            
            down = (pos[0]+1, pos[1])
            if down[0] <= 7 and map[down]==0 and down in empty_pos:
                empty_pos.remove(down)
                open_list.append(down)
            
            left = (pos[0], pos[1]-1)
            if left[1] >= 0 and map[left]==0 and left in empty_pos:
                empty_pos.remove(left)
                open_list.append(left)
            
            right = (pos[0], pos[1]+1)
            if right[1] <= 7 and map[right]==0 and right in empty_pos:
                empty_pos.remove(right)
                open_list.append(right)

            close_list.append(pos)


        partition_list.append(close_list)

    return partition_list
    


class Environment:
    def __init__(self, num_agents=config.num_agents, map_size=config.map_size):
        '''
        self.map:
            0 = empty
            1 = obstacle
        '''
        self.num_agents = num_agents
        self.map_size = map_size
        self.map = np.random.choice(2, self.map_size, p=[1-config.obstacle_density, config.obstacle_density]).astype(np.float32)
        partition_list = map_partition(self.map)
        partition_list = [ partition for partition in partition_list if len(partition) > 1 ]

        while not partition_list:
            self.map = np.random.choice(2, self.map_size, p=[1-config.obstacle_density, config.obstacle_density]).astype(np.float32)
            partition_list = map_partition(self.map)
            partition_list = [ partition for partition in partition_list if len(partition) >= 3 ]
        
        
        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int8)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=np.int8)
        
        for i in range(self.num_agents):
            partition = random.choice(partition_list)
            partition_list.remove(partition)

            pos = random.choice(partition)
            partition.remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int8)

            pos = random.choice(partition)
            partition.remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=np.int8)

            if len(partition) >= 3:
                partition_list.append(partition)

        self.steps = 0

    def reset(self):

        self.map = np.random.choice(2, self.map_size, p=[1-config.obstacle_density, config.obstacle_density]).astype(np.float32)
        partition_list = map_partition(self.map)
        partition_list = [ partition for partition in partition_list if len(partition) > 1 ]

        while not partition_list:
            self.map = np.random.choice(2, self.map_size, p=[1-config.obstacle_density, config.obstacle_density]).astype(np.float32)
            partition_list = map_partition(self.map)
            partition_list = [ partition for partition in partition_list if len(partition) >= 3 ]
        
        
        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int8)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=np.int8)
        
        for i in range(self.num_agents):
            partition = random.choice(partition_list)
            partition_list.remove(partition)

            pos = random.choice(partition)
            partition.remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int8)

            pos = random.choice(partition)
            partition.remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=np.int8)

            if len(partition) >= 3:
                partition_list.append(partition)

        self.steps = 0

        return self.observe()

    def load(self, world, num_agents, agents_pos, goals_pos):

        self.num_agents = num_agents
        self.map_size = (8, 8)
        self.map = np.array(world)
        self.goals_pos = np.array(goals_pos, dtype=np.int8)

        self.agents_pos = np.array(agents_pos, dtype=np.int8)

        
        self.steps = 0


    def step(self, actions: List):
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

        done = False
        
        if np.unique(self.agents_pos, axis=0).shape[0] < self.num_agents:
            print(self.steps)
            print(self.map)
            print(self.agents_pos)
            raise RuntimeError('unique')

        check_id = [i for i in range(self.num_agents)]

        rewards = [ None for _ in range(self.num_agents) ]

        # remove no movement agent id
        for agent_id in check_id.copy():

            if actions[agent_id] == 0:
                # stay
                check_id.remove(agent_id)

                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                    rewards[agent_id] = config.stay_on_goal_reward
                else:
                    rewards[agent_id] = config.stay_off_goal_reward
            else:
                # move
                rewards[agent_id] = config.move_reward


        next_pos = np.copy(self.agents_pos)

        for agent_id in check_id:
            next_pos[agent_id] += action_list[actions[agent_id]]




        for agent_id in check_id.copy():

            # move

            if np.any(next_pos[agent_id]<np.array([0,0])) or np.any(next_pos[agent_id]>=np.asarray(self.map_size)): 
                # agent out of bound
                rewards[agent_id] = config.collision_reward
                next_pos[agent_id] = self.agents_pos[agent_id]
                check_id.remove(agent_id)
                done = True

            elif self.map[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                rewards[agent_id] = config.collision_reward
                next_pos[agent_id] = self.agents_pos[agent_id]
                check_id.remove(agent_id)
                done = True

            
        

        flag = False
        while not flag:
            
            flag = True
            for agent_id in check_id.copy():
                
                if np.sum(np.all(next_pos==next_pos[agent_id], axis=1)) > 1:
                    # collide agent

                    collide_agent_id = np.where(np.all(next_pos==next_pos[agent_id], axis=1))[0].tolist()
                    collide_agent_id = [ id for id in collide_agent_id if id in check_id]
                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = config.collision_reward

                    for id in collide_agent_id:
                        check_id.remove(id)

                    flag = False
                    done = True
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
                        done = True
                        break


        self.agents_pos = np.copy(next_pos)

        self.steps += 1

        # check done
        if np.all(self.agents_pos==self.goals_pos):
            rewards = np.ones(self.num_agents) * config.finish_reward
            done = True
        elif self.steps >= config.max_steps:
            done = True

        print(rewards)
        print(done)
        return self.observe(), rewards, done, dict()


    def observe(self):
        obs = np.zeros((self.num_agents, 3, 8, 8), dtype=np.float32)
        for i in range(self.num_agents):
            obs[i,0][tuple(self.agents_pos[i])] = 1
            obs[i,1][tuple(self.goals_pos[i])] = 1
            obs[i,2,:,:] = np.copy(self.map)

        return obs

    
    def render(self):
        map = np.copy(self.map)
        for agent_id in range(self.num_agents):
            if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                map[tuple(self.agents_pos[agent_id])] = 4
            else:
                map[tuple(self.agents_pos[agent_id])] = 2
                map[tuple(self.goals_pos[agent_id])] = 3

        cmap = colors.ListedColormap(['white','grey','lime','purple','gold'])
        plt.imshow(map, cmap=cmap)
        plt.xlabel(str(self.steps))
        plt.ion()
        plt.show()
        plt.pause(0.5)


if __name__ == '__main__':
    a = np.array([[1,2],[3,4]])
    b = np.array([1,2])
    print(sum(np.all(a==b, axis=1)))
