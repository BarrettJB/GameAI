# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:32:05 2022

@author: bryso
"""

#example taken from https://towardsdatascience.com/creating-deep-neural-networks-from-scratch-an-introduction-to-reinforcement-learning-part-i-549ef7b149d2


import gym
import numpy as np

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
    
    self.layers = [NNLayer(input_size+1,self.hidden_size,activation=relu)]
    for i in range(self.num_hidden_layers-1):
      self.layers.append(NNLayer(self.hidden_size+1, self.hidden_size, activation=relu))
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
    return val
  
  def relu(mat):
    return np.multiply(mat,(mat>0))


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
      observation, reward, done, info = env.step(action)
      model.epsilon = model.epsilon if model.epsilon < 0.01 else model.epsilon*0.995
      if done:
        print('Episode {} ended after {} timesteps, current exploration is {}'.format(i_episode, t+1,model.epsilon))
        break
