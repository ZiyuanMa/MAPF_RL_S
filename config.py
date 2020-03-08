# environment setting
env_size = (10, 10)
num_agents = 2
obstacle_density = 0.15
action_space = 5
obs_dimension = 3

# reward setting
move_reward = -0.02
stay_on_goal_reward = 0
stay_off_goal_reward = -0.04
collision_reward = -0.1
finish_reward = 1

max_steps = 100

# model setting
num_kernels = 64
num_sa_heads = 4
num_sa_layers = 2


# training setting
greedy_coef = 1
checkpoint = 1000
training_eposide = 10
batch_size = 512

buffer_size = 32768


