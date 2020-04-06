import math


# environment setting
map_size = (8, 8)
num_agents = 2
obstacle_density = (0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4)
action_space = 5
obs_dimension = 3
max_steps = 50

# reward setting
move_reward = -0.075
stay_on_goal_reward = 0
stay_off_goal_reward = -0.125
collision_reward = -0.5
finish_reward = 3

# model setting
num_kernels = 128
num_sa_heads = 8
num_sa_layers = 2
latent_dim = 512


# training setting

forward_steps = 5


grad_norm = 10
batch_size = 128
double_q = True
buffer_size = 2000
exploration_fraction = 0.01
exploration_final_eps = 0.01
train_freq = 2000

learning_starts = 2000
save_interval = 200000
target_network_update_freq = 2000
gamma = 0.99
prioritized_replay = False
prioritized_replay_alpha = 0.6
prioritized_replay_beta0 = 0.4
param_noise = False
dueling = True
atom_num = 1
min_value = -5
max_value = 5
ob_scale = 1


imitation_ratio = 0.0