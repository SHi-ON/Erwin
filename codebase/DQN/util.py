import numpy as np

grid_size = 9  # 3 x 3
num_actions = 2  # [0, 10]
v1_max = 200
reward_col = 20

samples_name = "project_sample1"
# samples_name = "sample2"

policy_name = "policy_"

md_name = "dqn_model"
md_appendix_10_1 = "_10_1"
md_appendix_10_50 = "_10_50"
md_appendix_500_1 = "_500_1"
md_appendix_500_50 = "_500_50"
# model_name = md_name
# model_name = md_name + md_appendix_500_1
model_name = md_name + md_appendix_10_1


# hyper-parameters
gamma = 0.9  # discount factor
epsilon = .1  # epsilon-greedy exploration
alpha = .2  # learning rate

mem_len = 500  # Experience Replay length
batch_size = 1  # number of experiences per batch
# batch_size = 50  # number of experiences per batch
hidden_size = 100  # size of the hidden layers

col_names = ['V1', 'pop_1', 'pop_2', 'pop_3', 'pop_4', 'pop_5', 'pop_6',
                 'pop_7', 'pop_8', 'pop_9', 'sbank_1', 'sbank_2', 'sbank_3',
                 'sbank_4', 'sbank_5', 'sbank_6', 'sbank_7', 'sbank_8', 'sbank_9',
                 'actions', 'rewards']

policy_col_names = ['V1', 'pop_1', 'pop_2', 'pop_3', 'pop_4', 'pop_5', 'pop_6',
                 'pop_7', 'pop_8', 'pop_9', 'sbank_1', 'sbank_2', 'sbank_3',
                 'sbank_4', 'sbank_5', 'sbank_6', 'sbank_7', 'sbank_8', 'sbank_9',
                 'actions', 'rewards', 'policy']


def pick_action(q):
    idx = np.argmax(q)
    return 0 if idx == 0 else 10


def stopwatch_log(start, end):
    line = model_name + " for " + samples_name + " = " + str(np.round(end - start, 2)) + " seconds " + "\n"
    with open("./DQN/time_sheet.txt", "a") as time_file:
        time_file.write(line)


def plotter(df, idx, loss):
    if loss > 10e6:
        df.iloc[idx, 1] = 10e6


