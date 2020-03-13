import pickle
import random
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model, Sequential   
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam
import gym

#######################################################################################################
################################################# MLP #################################################
#######################################################################################################

class mlp():
    def __init__(self, s_size, a_size, e = 0.01, g = 0.95, memory_size = 2000, learning_rate = 0.001):
        self.s_size = s_size
        self.a_size = a_size
        self.e = e
        self.e_decay = 0.99
        self.g = g
        self.memory = deque(maxlen = memory_size)
        self.learning_rate = learning_rate
        self.q_model = self._build_model()
        self.target_model = self._build_model()
        self.copy_weights()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim = self.s_size, activation = 'relu'))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(32, activation = 'relu'))
        model.add(Dense(self.a_size, activation = 'linear'))
        model.compile(loss = "mean_squared_error", optimizer = Adam(lr = self.learning_rate))
        return model

    def copy_weights(self):
        self.target_model.set_weights(self.q_model.get_weights())

    def store_train_data(self, old_state, old_action, reward, state, is_done):
        self.memory.append((old_state, old_action, reward, state, is_done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for old_state, old_action, reward, state, is_done in minibatch:
            target = self.q_model.predict(old_state)
            if is_done:
                target[0][old_action] = reward
            else:
                a = self.q_model.predict(state)[0]
                t = self.target_model.predict(state)[0]
                target[0][old_action] = reward + self.g * t[np.argmax(a)]
            self.q_model.fit(old_state, target, epochs = 1, verbose = 0)

    def predict(self, state):
        if random.random() < self.e:
            return random.choice(range(self.a_size))
        act = self.q_model.predict(state)  
        return np.argmax(act[0])

#######################################################################################################
############################################ Agent Executing ##########################################
#######################################################################################################

env = gym.make('Acrobot-v1')
agent = mlp(env.obs_space.shape[0], env.action_space.n, e = 0.01)
batch_size = 32
rewards = []
steps = []

for i_episode in range(10):
    prev_obs = env.reset()
    old_action = agent.predict(np.reshape(prev_obs, [1, env.obs_space.shape[0]]))
    step = 0
    while True:
        step = step + 1
        #env.render()
        obs, reward, done, info = env.step(old_action)
        if done:
            if reward == 0:
                reward = 100
        agent.store_train_data(np.reshape(prev_obs, [1, env.obs_space.shape[0]]), old_action, reward, np.reshape(obs, [1, env.obs_space.shape[0]]), done)
        if len(agent.memory) > batch_size:
            agent.train(batch_size)
        prev_obs = obs
        old_action = agent.predict(np.reshape(obs, [1, env.obs_space.shape[0]]))
        if done:
            print("{}:{} steps: {}".format(i_episode, step, reward))
            rewards.append(1 if reward == 0 else 0)
            steps.append(step)
            agent.copy_weights()
            break

    score = sum(steps[-100:]) / 100
    if len(steps) >= 100 and score < 275:
        print("---------------------------------------------------------------")
        print("done")
        break
