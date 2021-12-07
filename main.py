import gym
import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt

from dqn_utils import Agent, eps_greedy


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env.seed(1)

    state_n = env.observation_space.shape[0]
    action_n = env.action_space.n

    #Parameters
    batch_size = 128
    num_episodes = 1000
    gamma = 0.998
    learning_rate = 0.001
    buffer_size = 10000

    eps = 1
    eps_max = 0.5
    eps_min = 0.1
    eps_decay = 200
    max_steps = 200

    max_avg_reward = float('-inf')
    total_reward = 0
    rewards = []
    rewards_list = deque(maxlen=100)

    agent = Agent(state_n, action_n, seed=1, learning_rate=learning_rate, gamma=gamma, buffer_size=buffer_size)

    for i in range(1, num_episodes+1):
        eps = eps_min + (eps_max - eps_min) * math.exp(-1. * i / eps_decay)
        state = np.array(env.reset())
        done = False
        total_reward = 0

        while not done:
            action = eps_greedy(agent, action_n, state, eps)

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            reward += 15*(abs(next_state[1]))
            next_state = np.array(next_state)
            agent.step((state, action, reward, next_state, done))
            state = next_state

        rewards.append(total_reward)
        rewards_list.append(total_reward)

        if i >= 100:
            avg_reward = np.mean(rewards_list)
            if avg_reward > max_avg_reward:
                max_avg_reward = avg_reward
                agent.save()
            if i % 100 == 0:
                print("Episode {}/{}, Max Average Score: {:.2f}, eps: {:.2f}".format(i, num_episodes, max_avg_reward, eps))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(rewards)), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    env.close()

