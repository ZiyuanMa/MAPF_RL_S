import math


# environment setting
env_size = (8, 8)
max_num_agents = 4
obstacle_density = 0.15
action_space = 5
obs_dimension = 3
max_steps = 100

# reward setting
move_reward = -0.02
stay_on_goal_reward = -0.02
stay_off_goal_reward = -0.02
collision_reward = -0.2
finish_reward = 1

# model setting
num_kernels = 32
num_sa_heads = 4
num_sa_layers = 2


# training setting
training_steps = 300
checkpoint = training_steps // 5
update_steps = 10
batch_size = 50000
mini_batch_size = 50
buffer_size = 500000
search_tree_depth = math.ceil(math.log2(buffer_size))-4
TD_steps = 5
gamma = 0.99




