import torch
import numpy as np
#import cupy as cp
import time
import random
from running_mean_std import RunningMeanStd
from test import evaluate_model
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


class Train:
    def __init__(self, env, test_env, env_name, n_iterations, agent, epochs, mini_batch_size, epsilon, horizon):
        self.env = env
        self.env_name = env_name
        self.test_env = test_env
        self.agent = agent
        self.epsilon = epsilon
        self.horizon = horizon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.n_iterations = n_iterations

        self.start_time = 0
        self.state_rms = RunningMeanStd(shape=(self.agent.n_states,))

        self.running_reward = 0
        #finetuning
        self.state_rms_mean, self.state_rms_var = self.agent.load_weights_f()

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs, values, log_probs):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[indices], actions[indices], returns[indices], advs[indices], values[indices],\
                  log_probs[indices]

    def train(self, states, actions, advs, values, log_probs):
    
        values = np.vstack(values[:-1])
        #log_probs = cp.vstack(log_probs)
        log_probs = np.vstack(log_probs)
        returns = advs + values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        actions = np.vstack(actions)
        for epoch in range(self.epochs):
            for state, action, return_, adv, old_value, old_log_prob in self.choose_mini_batch(self.mini_batch_size,
                                                                                               states, actions, returns,
                                                                                               advs, values, log_probs):
                state = torch.Tensor(state).to(self.agent.device)
                action = torch.Tensor(action).to(self.agent.device)
                return_ = torch.Tensor(return_).to(self.agent.device)
                adv = torch.Tensor(adv).to(self.agent.device)
                old_value = torch.Tensor(old_value).to(self.agent.device)
                old_log_prob = torch.Tensor(old_log_prob).to(self.agent.device)

                value = self.agent.critic(state)
                # clipped_value = old_value + torch.clamp(value - old_value, -self.epsilon, self.epsilon)
                # clipped_v_loss = (clipped_value - return_).pow(2)
                # unclipped_v_loss = (value - return_).pow(2)
                # critic_loss = 0.5 * torch.max(clipped_v_loss, unclipped_v_loss).mean()
                critic_loss = self.agent.critic_loss(value, return_)

                new_log_prob = self.calculate_log_probs(self.agent.current_policy, state, action)

                ratio = (new_log_prob - old_log_prob).exp()
                actor_loss = self.compute_actor_loss(ratio, adv)

                self.agent.optimize(actor_loss, critic_loss)

        return actor_loss, critic_loss

    def step(self):
        state = self.env.reset()
        k = np.array([1,1,1,1,1,1,1,1],dtype='float64')
        k1 = np.array([1,1,1,1,1,1,1,1],dtype='float64')#base
        df = pd.read_csv("attack05_15_j_train.csv")
        for iteration in range(1, 1 + self.n_iterations):
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []
            #attackmake
            seed=time.time()#random
            rate=random.random()
            if rate >= 0.7:
                #print("a")
                df1 = df.sample()
                k=df1.to_numpy()
                #print(k)
                #curriculum
                #k=k1+(k-k1)*(iteration/self.n_iterations)
            else:
                k = np.array([1,1,1,1,1,1,1,1],dtype='float64')
                #print("b")

            self.start_time = time.time()
            for t in range(self.horizon):
                # self.state_rms.update(state)
                state = np.clip((state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
                dist = self.agent.choose_dist(state)
                action = dist.sample().cpu().numpy()[0]*k
                # action = np.clip(action, self.agent.action_bounds[0], self.agent.action_bounds[1])
                log_prob = dist.log_prob(torch.Tensor(action).to(self.agent.device))
                value = self.agent.get_value(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done)

                if done:
                    state = self.env.reset()
                else:
                    state = next_state
            # self.state_rms.update(next_state)
            next_state = np.clip((next_state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
            next_value = self.agent.get_value(next_state) * (1 - done)
            values.append(next_value)

            advs = self.get_gae(rewards, values, dones)
            states = np.vstack(states)
            actor_loss, critic_loss = self.train(states, actions, advs, values, log_probs)
            # self.agent.set_weights()
            self.agent.schedule_lr()
            eval_rewards = evaluate_model(self.agent, self.test_env, self.state_rms, self.agent.action_bounds)
            self.state_rms.update(states)
            self.print_logs(iteration, actor_loss, critic_loss, eval_rewards)

    @staticmethod
    def get_gae(rewards, values, dones, gamma=0.99, lam=0.95):#gamma=0.99

        advs = []
        gae = 0

        dones.append(0)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * (values[step + 1]) * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advs.append(gae)

        advs.reverse()
        return np.vstack(advs)

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution = model(states)
        return policy_distribution.log_prob(actions)

    def compute_actor_loss(self, ratio, adv):
        pg_loss1 = adv * ratio
        pg_loss2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(pg_loss1, pg_loss2).mean()
        return loss

    def print_logs(self, iteration, actor_loss, critic_loss, eval_rewards):
        if iteration == 1:
            self.running_reward = eval_rewards
        else:
            self.running_reward = self.running_reward * 0.99 + eval_rewards * 0.01

        if iteration % 100 == 0:
            print(f"Iter:{iteration}| "
                  f"Ep_Reward:{eval_rewards:.3f}| "
                  f"Running_reward:{self.running_reward:.3f}| "
                  f"Actor_Loss:{actor_loss:.3f}| "
                  f"Critic_Loss:{critic_loss:.3f}| "
                  f"Iter_duration:{time.time() - self.start_time:.3f}| "
                  f"lr:{self.agent.actor_scheduler.get_last_lr()}")
            self.agent.save_weights(iteration, self.state_rms)

        """with SummaryWriter(self.env_name + "/logs") as writer:
            writer.add_scalar("Episode running reward", self.running_reward, iteration)
            writer.add_scalar("Episode reward", eval_rewards, iteration)
            writer.add_scalar("Actor loss", actor_loss, iteration)
            writer.add_scalar("Critic loss", critic_loss, iteration)"""
