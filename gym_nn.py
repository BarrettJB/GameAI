# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:32:05 2022

@author: bryso
"""

#example taken from https://towardsdatascience.com/creating-deep-neural-networks-from-scratch-an-introduction-to-reinforcement-learning-part-i-549ef7b149d2
#https://towardsdatascience.com/creating-deep-neural-networks-from-scratch-an-introduction-to-reinforcement-learning-6bba874019db

import gym
import numpy as np
from collections import deque

class NNLayer:
  def __init__(self, input_size, output_size, activation=None, lr = 0.001):
    self.input_size = input_size
    self.output_size = output_size
    self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size, output_size))
    self.activation_function = activation()
    self.lr = lr
    
  def forward(self, inputs, remember_for_backprop=True):
    input_with_bias = np.append(np.ones((len(inputs),1)),inputs, axis=1)
    unactivated = np.dot(input_with_bias,self.weights)
    output = unactivated
    if self.activation_function != None:
      output = self.activation_function(output)
    if remember_for_backprop:
      self.backward_store_in = input_with_bias
      self.backward_store_out = np.copy(unactivated)
      
    return output

class RLAgent:
  env = None
  
  def __init__(self, env):
    self.env = env
    self.hidden_size = 24
    self.input_size = env.observation_space.shape[0]
    self.output_size = env.action_space.n
    self.num_hidden_layers = 2
    self.epsilon = 1.0
    self.memory = deque([],1000000)
    
    self.layers = [NNLayer(self.input_size+1,self.hidden_size,activation=self.relu)]
    for i in range(self.num_hidden_layers-1):
      self.layers.append(NNLayer(self.hidden_size+1, self.hidden_size, activation=self.relu))
    self.layers.append(NNLayer(self.hidden_size+1,self.output_size))
    
  def select_action(self, observation):
    values = self.foward(observation)
    if (np.random.random() > self.epsilon):
      return np.argmax(values)
    else:
      return np.random.randint(self.env.action_space.n)
    
  def forward(self, observation, remember_for_backprop=True):
    vals = np.copy(observation)
    index = 0
    for layer in self.layers:
      vals = layer.forward(vals, remember_for_backprop)
      index += 1
    return vals
  
  def relu(mat):
    return np.multiply(mat,(mat>0))
  
  def remember(self, done, action, observation, prev_obs):
    self.memory.append([done, action, observation, prev_obs])
    
  def experience_replay(self, update_size=20):
    if (len(self.memory) < update_size):
      return
    else:
      batch_indices = np.random.choice(len(self.memory), update_size)
      for idx in batch_indices:
        done, action_selected, new_obs, prev_obs = self.memory[idx]
        action_values = self.forward(prev_obs, remember_for_backprop=True)
        next_action_values = self.forward(new_obs, remember_for_backprop=False)
        experimental_values = np.copy(action_values)
        if done:
          experimental_values[action_selected] = -1
        else:
          experimental_values[action_selected] = 1 + self.gamma*np.max(next_action_values)
        self.backward(action_values, experimental_values)
        self.epsilon = self.epsilon if self.epsilon < 0.01 else self.epsilon*0.995
        for layer in self.layers:
          layer.lr - layer.lr if layer.lr < 0.0001 else layer.lr*0.995
        


if __name__ == '__main__':
  env = gym.make('CartPole-v1')
  NUM_EPISODES = 10
  MAX_TIMESTEPS = 1000
  model = RLAgent(env)
  
  for i in range(NUM_EPISODES):
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
      env.render()
      action = model.select_action(observation)
      prev_obs = observation
      observation, reward, done, info = env.step(action)
      model.remember(done, action, observation, prev_obs)
      model.experience_replay(20)
      model.epsilon = model.epsilon if model.epsilon < 0.01 else model.epsilon*0.995
      if done:
        print('Episode {} ended after {} timesteps, current exploration is {}'.format(i_episode, t+1,model.epsilon))
        break
