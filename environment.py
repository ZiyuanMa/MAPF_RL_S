import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from numba import jit
import config
import random

from typing import List

action_list = np.array([[0, 0],[-1, 0],[1, 0],[0, -1],[0, 1]], dtype=np.int8)


def map_partition(map):

    empty_pos = np.argwhere(map==0).astype(np.int).tolist()

    empty_pos = [ tuple(pos) for pos in empty_pos]

    if not empty_pos:
        raise RuntimeError('no empty position')

    partition_list = list()
    while empty_pos:

        start_pos = empty_pos.pop()

        open_list = list()
        open_list.append(start_pos)
        close_list = list()

        while open_list:
            pos = open_list.pop(0)

            up = (pos[0]-1, pos[1])
            if up[0] >= 0 and map[up]==0 and up in empty_pos:
                empty_pos.remove(up)
                open_list.append(up)
            
            down = (pos[0]+1, pos[1])
            if down[0] < map.shape[0] and map[down]==0 and down in empty_pos:
                empty_pos.remove(down)
                open_list.append(down)
            
            left = (pos[0], pos[1]-1)
            if left[1] >= 0 and map[left]==0 and left in empty_pos:
                empty_pos.remove(left)
                open_list.append(left)
            
            right = (pos[0], pos[1]+1)
            if right[1] < map.shape[1] and map[right]==0 and right in empty_pos:
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
        self.obstacle_density = random.choice(config.obstacle_density)
        self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)
        partition_list = map_partition(self.map)
        partition = sorted(partition_list, key= lambda x: len(x), reverse=True)[0]
        # partition_list = [ partition for partition in partition_list if len(partition) >= 4 ]

        while len(partition) < self.num_agents*3:
            self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)
            partition_list = map_partition(self.map)
            partition = sorted(partition_list, key= lambda x: len(x), reverse=True)[0]
            # partition_list = [ partition for partition in partition_list if len(partition) >= 4 ]
        
        
        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int8)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=np.int8)
        
        for i in range(self.num_agents):

            pos = random.choice(partition)
            partition.remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int8)

            pos = random.choice(partition)
            partition.remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=np.int8)

        self.steps = 0

    def reset(self):

        self.obstacle_density = random.choice(config.obstacle_density)
        self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)
        partition_list = map_partition(self.map)
        partition = sorted(partition_list, key= lambda x: len(x), reverse=True)[0]
        # partition_list = [ partition for partition in partition_list if len(partition) >= 4 ]

        while len(partition) < self.num_agents*3:
            self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)
            partition_list = map_partition(self.map)
            partition = sorted(partition_list, key= lambda x: len(x), reverse=True)[0]
            # partition_list = [ partition for partition in partition_list if len(partition) >= 4 ]
        
        
        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int8)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=np.int8)
        
        for i in range(self.num_agents):

            pos = random.choice(partition)
            partition.remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int8)

            pos = random.choice(partition)
            partition.remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=np.int8)

        self.steps = 0

        return self.observe()

    def load(self, world: np.ndarray, agents_pos: np.ndarray, goals_pos: np.ndarray):

        self.map = np.copy(world)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)

        assert agents_pos.shape[0] == self.num_agents
        
        self.steps = 0

    def step(self, actions: List[int]):
        '''
        actions:
            list of indices
                0 stay
                1 up
                2 down
                3 left
                4 right
        '''

        assert len(actions) == self.num_agents, 'actions number' + str(actions)
        # assert all([action_idx<config.action_space and action_idx>=0 for action_idx in actions]), 'action index out of range'
        
        if np.unique(self.agents_pos, axis=0).shape[0] < self.num_agents:
            print(self.steps)
            print(self.map)
            print(self.agents_pos)
            raise RuntimeError('unique')

        check_id = [i for i in range(self.num_agents)]

        rewards = np.empty(self.num_agents, dtype=np.float32)

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

            elif self.map[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                rewards[agent_id] = config.collision_reward
                next_pos[agent_id] = self.agents_pos[agent_id]
                check_id.remove(agent_id)



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

        if np.all(self.agents_pos==self.goals_pos):
            rewards = np.ones(self.num_agents, dtype=np.float32) * config.finish_reward
            done = True
        elif self.steps >= config.max_steps:
            done = True
        else:
            done = False


        return self.observe(), rewards, done, dict()


    def observe(self):
        obs = np.zeros((self.num_agents, 3, 8, 8), dtype=np.float32)
        for i in range(self.num_agents):
            obs[i,0][tuple(self.agents_pos[i])] = 1
            obs[i,1][tuple(self.goals_pos[i])] = 1
            obs[i,2,:,:] = np.copy(self.map==0)

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

    def close(self):
        plt.close()


