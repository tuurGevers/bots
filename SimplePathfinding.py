import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import random
import json
import time

# Define the grid environment
class GridEnv:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.start = (0, 0)
        self.goal = (self.grid_size - 1, self.grid_size - 1)
        self.position = self.start
        return self._get_state()

    def _get_state(self):
        state = np.copy(self.grid)
        state[self.position] = 2
        state[self.goal] = 3
        return state.flatten()

    def step(self, action):
        x, y = self.position
        if action == 0 and x > 0: x -= 1  # up
        if action == 1 and x < self.grid_size - 1: x += 1  # down
        if action == 2 and y > 0: y -= 1  # left
        if action == 3 and y < self.grid_size - 1: y += 1  # right
        self.position = (x, y)
        done = self.position == self.goal
        reward = 1 if done else -0.1
        return self._get_state(), reward, done

# Define the neural network model
def build_model(input_size, output_size):
    model = Sequential([
        Flatten(input_shape=(input_size,)),
        Dense(24, activation='relu'),
        Dense(24, activation='relu'),
        Dense(output_size, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the model using Q-learning
def train_model(env, model, episodes, max_steps, epsilon, gamma, batch_size, training_time):
    memory = []
    start_time = time.time()
    for e in range(episodes):
        if time.time() - start_time > training_time:  # Stop training after the specified time
            break
        state = env.reset()
        total_reward = 0
        for time_step in range(max_steps):
            if np.random.rand() <= epsilon:
                action = random.randrange(4)
            else:
                q_values = model.predict(state.reshape(1, -1))
                action = np.argmax(q_values[0])
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if done:
                break
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                for s, a, r, ns, d in minibatch:
                    target = r
                    if not d:
                        target += gamma * np.amax(model.predict(ns.reshape(1, -1))[0])
                    target_f = model.predict(s.reshape(1, -1))
                    target_f[0][a] = target
                    model.fit(s.reshape(1, -1), target_f, epochs=1, verbose=0)
        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2}")
        if epsilon > 0.01:
            epsilon *= 0.995

# Parameters
grid_size = 5
input_size = grid_size * grid_size
output_size = 4  # up, down, left, right
episodes = 1000 
max_steps = 100
epsilon = 1.0
gamma = 0.95
batch_size = 32
training_time = 300  # 5 minutes

# Create the environment and model
env = GridEnv(grid_size)
model = build_model(input_size, output_size)

# Train the model
train_model(env, model, episodes, max_steps, epsilon, gamma, batch_size, training_time)

# Save the model architecture and weights
model_json = model.to_json()
with open("pathfinding_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("pathfinding_model.weights.h5")

# Convert the weights to a serializable format
weights = model.get_weights()
weights_serializable = [w.tolist() for w in weights]
with open("pathfinding_model_weights_serializable.json", "w") as json_file:
    json.dump(weights_serializable, json_file)
