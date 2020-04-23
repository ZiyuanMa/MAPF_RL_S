# environment setting
map_size = (9, 9)
num_agents = 2
obstacle_density = (0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4)
action_space = 5
obs_dimension = 4
max_steps = 50

# reward setting
move_reward = -0.075
stay_on_goal_reward = 0
stay_off_goal_reward = -0.125
collision_reward = -0.5
finish_reward = 3

# model setting
num_kernels = 128
latent_dim = 512

# training setting

n_steps = 5


grad_norm = 10
batch_size = 64
double_q = True
buffer_size = 50000
exploration_start_eps = 1.0
exploration_final_eps = 0.01
train_freq = 8

learning_starts = 30000
save_interval = 20000
target_network_update_freq = 1000*train_freq
gamma = 0.99
prioritized_replay = True
prioritized_replay_alpha = 0.6
prioritized_replay_beta0 = 0.4
dueling = True

imitation_ratio = 0.25

history_steps = 3