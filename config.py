# environment setting
env_size = (8, 8)
max_num_agents = 4
obstacle_density = 0.15
action_space = 5
obs_dimension = 3
max_steps = 100

# reward setting
move_reward = -0.02
stay_on_goal_reward = 0
stay_off_goal_reward = -0.05
collision_reward = -0.1
finish_reward = 1

# model setting
num_kernels = 64
num_sa_heads = 4
num_sa_layers = 2


# training setting
greedy_coef = 1
checkpoint = 1000000
training_timesteps = 10000000
batch_size = 2000
buffer_size = 1000000
forward_steps = 3
gamma = 0.99



