#!/usr/bin/env python3
import random
import gym
import numpy as np
# import tensorflow.keras as keras
from collections import deque, namedtuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class Cartpole:
    '''
        Class CartPole - DQN agent for CartPole
    '''
    def __init__(self, total_intput, total_output):
        # ------ hyper parameter ---------

        self.gamma = 0.995
        self.learningRate = 0.0005
        self.memory_size = 1000000
        self.batch_size = 40
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.decay_rate = 0.995

        #---------------------------------
        
        self.epsilson = self.epsilon_max
        self.total_output = total_output
        self.batch = namedtuple("Batch", "state action reward next_state done")
        self.batches = deque(maxlen=self.memory_size)

        # Creating keras neural network
        self.model = Sequential()
        self.model.add(Dense(48, input_shape=(total_intput,), activation="relu"))
        self.model.add(Dense(48, activation="relu"))
        self.model.add(Dense(self.total_output, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.learningRate))

    def store(self, state, action, reward, next_state, done):
        self.batches.append(self.batch(state, action, reward, next_state, 1-done))

    def predict(self, state):
        if np.random.rand() < self.epsilson:
            return random.randrange(self.total_output)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def trainNN(self):
        # if number of training sample is less than batch size then return
        if len(self.batches) < self.batch_size:
            return

        # take random samples od bathc size
        batch = random.sample(self.batches, self.batch_size)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        states = np.vstack([i.state for i in batch if i is not None])
        actions = np.vstack([i.action for i in batch if i is not None])
        rewards = np.vstack([i.reward for i in batch if i is not None])
        next_states = np.vstack([i.next_state for i in batch if i is not None])
        dones = np.vstack([i.done for i in batch if i is not None])
        oldQ = self.model.predict(states)
        newQ = self.model.predict(next_states)
        q_values = oldQ.copy()
        tempQ = rewards + self.gamma * \
                np.amax(newQ, axis=1).reshape(self.batch_size,1)*dones
        q_values[batch_index, actions[:,0]] = tempQ[:,0]
        self.model.fit(states, q_values, verbose = 0)

        # update epsilon
        if self.epsilson * self.decay_rate < self.epsilon_min:
            self.epsilson = self.epsilon_min
        else:
            self.epsilson = self.epsilson * self.decay_rate

def train(agent, total_episodes, total_intput, total_output, env):
    '''
        train function - to train the neural network
    '''
    run = 0
    score = []
    no_of_episodes = []

    # Train for total_episodes
    for e in range(total_episodes):
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, total_intput])
        step = 0

        # Simulate time
        for time in range(500):
            step += 1
            env.render()
            # get action from agent
            action = agent.predict(state)
            # get next state and reward from enviromnment
            state_next, reward, done, _ = env.step(action)
            # If episode is terminated then neagte the reward
            # reward = reward if (not done) or time > 495 else -1*reward
            state_next = np.reshape(state_next, [1, total_intput])
            # Store the value for training later on
            agent.store(state, action, reward, state_next, done)
            state = state_next
            # train the network
            agent.trainNN()
            if done:
                print("Run: " + str(run) + ", exploration: " + str(agent.epsilson) + ", score: " + str(step))
                break
        score.append(step)
        no_of_episodes.append(e)
    return no_of_episodes, score

def plot(no_of_episodes, score):
    '''
        Plot function to plot the results
    '''
    plt.plot(no_of_episodes, score)
    plt.ylabel('Cummulative reward')
    plt.xlabel('Number of Episode')
    plt.savefig('Result.png')
    plt.show()
    return 0

def main():
    '''
        Main function
    '''
    env = gym.make("CartPole-v1")
    total_intput = env.observation_space.shape[0]
    total_output = env.action_space.n
    # create the agent
    agent = Cartpole(total_intput, total_output)
    total_episodes = 200
    # train the agent
    no_of_episodes, score = train(agent, total_episodes, total_intput, total_output, env)
    # plot the performance
    plot(no_of_episodes, score)

if __name__ == "__main__":
   main()
