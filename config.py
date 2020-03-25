import math


# environment setting
map_size = (8, 8)
num_agents = 2
obstacle_density = (0, 0.1, 0.2, 0.3)
action_space = 5
obs_dimension = 3
max_steps = 50

# reward setting
move_reward = -0.075
stay_on_goal_reward = 0
stay_off_goal_reward = -0.125
collision_reward = -0.5
finish_reward = 5

# model setting
num_kernels = 32
num_sa_heads = 4
num_sa_layers = 2


# training setting

forward_steps = 5


grad_norm = 10
batch_size = 32
double_q = True
buffer_size = 10000
exploration_fraction = 0.1
exploration_final_eps = 0.01
train_freq = 4
learning_starts = 10000
save_interval = 500000
target_network_update_freq = 1000
gamma = 0.99
prioritized_replay = True
prioritized_replay_alpha = 0.6
prioritized_replay_beta0 = 0.4
param_noise = False
dueling = True
atom_num = 51
min_value = -5
max_value = 5
ob_scale = 1


imitation_ratio = 0.2