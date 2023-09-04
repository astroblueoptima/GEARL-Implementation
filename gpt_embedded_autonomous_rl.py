
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.agent_position = [0, 0]
        self.goal_position = [size - 1, size - 1]
        self.move_reward = -1
        self.goal_reward = 10
        self.wall_penalty = -5

    def reset(self):
        self.agent_position = [0, 0]
        return self.agent_position

    def step(self, action):
        if action == 0:
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 1:
            self.agent_position[1] = min(self.size - 1, self.agent_position[1] + 1)
        elif action == 2:
            self.agent_position[0] = min(self.size - 1, self.agent_position[0] + 1)
        elif action == 3:
            self.agent_position[1] = max(0, self.agent_position[1] - 1)
        if self.agent_position == self.goal_position:
            return self.agent_position, self.goal_reward, True
        else:
            return self.agent_position, self.move_reward, False

class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        self.q_table = np.zeros((env.size, env.size, 4))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)
        else:
            return np.argmax(self.q_table[state[0], state[1], :])

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                old_value = self.q_table[state[0], state[1], action]
                future_max = np.max(self.q_table[next_state[0], next_state[1], :])
                new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * future_max)
                self.q_table[state[0], state[1], action] = new_value
                state = next_state
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# To run:
# env = GridWorld()
# agent = QLearning(env)
# agent.train()
