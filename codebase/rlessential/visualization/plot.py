import matplotlib.pyplot as plt
import numpy as np

import consts

# slides theme colors
theme_blue = '#0C2B36'
theme_red = '#E04D4F'
theme_green = '#00F900'

titles = {'original': 'Original reward', 'aggregate': 'Aggregate reward', 'adapted': 'Adapted aggregate reward'}


def average_rolling(array):
    """
    Calculates moving(rolling) average.

    :param array: input array
    :return: calculated cumulative moving average
    """
    cumulative_sum = array.cumsum()
    a = np.arange(array.size)
    a = a + 1
    rolling_mean = cumulative_sum / a
    return rolling_mean


def plot_steps(original, aggregate, adapted):
    fig, ax = plt.subplots()
    ax.set_xticks([], [])
    ax.set_yticks([], [])

    ax1 = fig.add_subplot(311)
    ax1.plot(original, color=theme_blue, label=titles['original'], lw=2)
    ax1.set_title(titles['original'])
    ax1.set_xticks([], [])

    ax2 = fig.add_subplot(312)
    ax2.plot(aggregate, color=theme_red, label=titles['aggregate'], lw=2)
    ax2.set_title(titles['aggregate'])
    ax2.set_xticks([], [])
    ax2.set_ylabel('rewards')

    ax3 = fig.add_subplot(313)
    ax3.plot(adapted, color=theme_green, label=titles['adapted'], lw=2)
    ax3.set_title(titles['adapted'])

    plt.xlabel('steps')
    return plt


def plot_cumulative(original, aggregate, adapted):
    plt.figure()
    plt.plot(original.cumsum(),
             color=theme_blue, label=titles['original'], lw=2)
    plt.plot(aggregate.cumsum(),
             color=theme_red, label=titles['aggregate'], lw=2)
    plt.plot(adapted.cumsum(),
             color=theme_green, label=titles['adapted'], lw=2)
    plt.title('Cumulative Reward')
    plt.xlabel('steps')
    plt.ylabel('rewards')
    plt.legend()
    return plt


def plot_rolling(original, aggregate, adapted):
    rolling_original = average_rolling(original)
    rolling_aggregate = average_rolling(aggregate)
    rolling_adapted = average_rolling(adapted)

    plt.figure()
    plt.plot(rolling_original, color=theme_blue, label=titles['original'], lw=2)
    plt.plot(rolling_aggregate, color=theme_red, label=titles['aggregate'], lw=2)
    plt.plot(rolling_adapted, color=theme_green, label=titles['adapted'], lw=2)
    plt.title('Rolling Average Reward')
    plt.xlabel('steps')
    plt.ylabel('rewards')
    plt.legend()
    return plt


def visualize_rewards(rewards, rewards_aggregate, rewards_aggregate_adapted, method_name):
    plot_steps(rewards, rewards_aggregate, rewards_aggregate_adapted)
    plt.savefig(consts.RES_PATH + method_name + '_' + 'steps_rewards.pdf', format='pdf')
    plt.show()

    plot_cumulative(rewards, rewards_aggregate, rewards_aggregate_adapted)
    plt.savefig(consts.RES_PATH + method_name + '_' + 'cumulative_rewards.pdf', format='pdf')
    plt.show()

    plot_rolling(rewards, rewards_aggregate, rewards_aggregate_adapted)
    plt.savefig(consts.RES_PATH + method_name + '_' + 'rolling_rewards.pdf', format='pdf')
    plt.show()
