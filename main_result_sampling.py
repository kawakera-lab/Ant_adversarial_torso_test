import gym
import os
import mujoco_py

from agent import Agent
from train import Train
from play import Play

import numpy as np
import statistics
import math
import time
import random
import pandas as pd
import copy
ENV_NAME = "Ant"
TRAIN_FLAG = False
#TRAIN_FLAG = True
test_env = gym.make(ENV_NAME + "-v2")# 実行する課題を設定
n_states = test_env.observation_space.shape[0]# 課題の状態と行動の数を設定
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
n_actions = test_env.action_space.shape[0]# 行動を取得
n_iterations = 20000

lr = 3e-5
epochs = 10
clip_range = 0.1
mini_batch_size = 256
T = 2048

def func_nomal(t):
	rewardlog=[]
	k = np.array([1,1,1,1,1,1,1,1],dtype='float64')
	for j in range(t):#15
		rewardlog.append(player.evaluate(k))
		#print(k,rewardlog)
	return statistics.mean(rewardlog),statistics.stdev(rewardlog),rewardlog
def func_random(t):
	rewardlog=[]
	df = pd.read_csv("ant_random_parameters_test.csv")
	for j in range(t):#15
		df1 = df.sample()
		k=df1.to_numpy()
		
		rewardlog.append(player.evaluate(k))
		#print(k,rewardlog)
	return statistics.mean(rewardlog),statistics.stdev(rewardlog),rewardlog
def func_adv(t):
	rewardlog=[]
	df = pd.read_csv("attack_adversarial_test.csv")
	for j in range(t):#15
		df1 = df.sample()
		k=df1.to_numpy()
		
		rewardlog.append(player.evaluate(k))
		#print(k,rewardlog)
	return statistics.mean(rewardlog),statistics.stdev(rewardlog),rewardlog
"""	
def func_nomal(t):
	rewardlog=[]
	k = np.array([1,1,1,1,1,1,1,1],dtype='float64')
	#k = np.array([1,1,1,1,1,1,1,1,1,1,1,1],dtype='float64')
	for j in range(t):#15
		rewardlog.append(player.evaluate(k))
	return statistics.mean(rewardlog),statistics.stdev(rewardlog),rewardlog
"""

if __name__ == "__main__":
    print(f"number of states:{n_states}\n"
          f"action bounds:{action_bounds}\n"
          f"number of actions:{n_actions}")

    if not os.path.exists(ENV_NAME):
        os.mkdir(ENV_NAME)
        os.mkdir(ENV_NAME + "/logs")

    env = gym.make(ENV_NAME + "-v2")
    env.seed(100)

    agent = Agent(n_states=n_states,
                  n_iter=n_iterations,
                  env_name=ENV_NAME,
                  action_bounds=action_bounds,
                  n_actions=n_actions,
                  lr=lr)
    if TRAIN_FLAG:
        trainer = Train(env=env,
                        test_env=test_env,
                        env_name=ENV_NAME,
                        agent=agent,
                        horizon=T,
                        n_iterations=n_iterations,
                        epochs=epochs,
                        mini_batch_size=mini_batch_size,
                        epsilon=clip_range)
        trainer.step()
        
    modelname = "Ant_advF_lr5_10_30000_weights"
    col=["reward"]
    player = Play(env, agent, ENV_NAME,10000)
    
    avr,hensa,dataset=func_nomal(1000)
    print(modelname +" in Nomal",avr,hensa)
    df_nomal = pd.DataFrame(dataset, columns=col)
    df_nomal.to_csv('env_comp_test/'+modelname +'_testNomal.csv',index=False)
    

    avr,hensa,dataset=func_random(1000)
    print(modelname +" in Random",avr,hensa)
    df_random = pd.DataFrame(dataset, columns=col)
    df_random.to_csv('env_comp_test/'+modelname +'_testRandom.csv',index=False)
    
    avr,hensa,dataset=func_adv(1000)
    print(modelname +" in Adversarial",avr,hensa)
    df_adv = pd.DataFrame(dataset, columns=col)
    df_adv.to_csv('env_comp_test/'+modelname +'_testAdv.csv',index=False)
    
    
    

