# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:32:05 2022

@author: bryso
"""

#example taken from https://towardsdatascience.com/creating-deep-neural-networks-from-scratch-an-introduction-to-reinforcement-learning-part-i-549ef7b149d2
#https://towardsdatascience.com/creating-deep-neural-networks-from-scratch-an-introduction-to-reinforcement-learning-6bba874019db

#further resources
#http://www.briandolhansky.com/blog/2013/9/27/artificial-neural-networks-backpropagation-part-4

import gym
import numpy as np
from collections import deque

def relu(mat):
  return np.multiply(mat,(mat>0))

def relu_derivative(mat):
  return (mat>0)*1

class NNLayer:
  def __init__(self, input_size, output_size, activation=None, lr = 0.001):
    self.input_size = input_size
    self.output_size = output_size
    self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size, output_size))
    self.activation_function = activation
    self.lr = lr

  def update_weights(self, gradient):
    self.weights = self.weights = self.lr*gradient  
  
  def forward(self, inputs, remember_for_backprop=True):
    input_with_bias = np.append(inputs,1)
    unactivated = np.dot(input_with_bias,self.weights)
    output = unactivated
    if self.activation_function != None:
      output = self.activation_function(output)
    if remember_for_backprop:
      self.backward_store_in = input_with_bias
      self.backward_store_out = np.copy(unactivated)
    return output
      
  def backward(self, gradient):
    adjusted_mul = gradient
    if self.activation_function != None:
      adjusted_mul = np.multiply(relu_derivative(self.backward_store_out), gradient)
    D_i = np.dot(np.transpose(np.reshape(self.backward_store_in,(1,len(self.backward_store_in)))), np.reshape(adjusted_mul,(1,len(adjusted_mul))))
    delta_i = np.dot(adjusted_mul, np.transpose(self.weights))[:-1]
    self.update_weights(D_i)
    return delta_i

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
    self.gamma = 0.95
    
    self.layers = [NNLayer(self.input_size+1,self.hidden_size,activation=relu)]
    for i in range(self.num_hidden_layers-1):
      self.layers.append(NNLayer(self.hidden_size+1, self.hidden_size, activation=relu))
    self.layers.append(NNLayer(self.hidden_size+1,self.output_size))
    
  def select_action(self, observation):
    values = self.forward(observation)
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
  
  def backward(self, calculated_values, experimental_values):
    delta = (calculated_values - experimental_values)
    for layer in reversed(self.layers):
      delta = layer.backward(delta)
  
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
        self.epsilon = self.epsilon if self.epsilon < 0.01 else self.epsilon*0.999
        for layer in self.layers:
          layer.lr - layer.lr if layer.lr < 0.0001 else layer.lr*0.995
        


if __name__ == '__main__':
  env = gym.make('CartPole-v1')
  NUM_EPISODES = 5000
  MAX_TIMESTEPS = 1000
  model = RLAgent(env)
  
  perf_tracker = np.zeros(25)

  for i in range(NUM_EPISODES):
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
      env.render()
      action = model.select_action(observation)
      prev_obs = observation
      observation, reward, done, info = env.step(action)
      model.remember(done, action, observation, prev_obs)
      model.experience_replay(20)
      if done:
        perf_tracker[i%25] = t+1
        print('Episode {} ended after {} timesteps, current exploration is {}, avg run is {}'.format(i, t+1,model.epsilon,np.average(perf_tracker)))
        break
