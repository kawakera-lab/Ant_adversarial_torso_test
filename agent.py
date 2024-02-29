from model import Actor, Critic
from torch.optim import Adam
from torch import from_numpy
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ExponentialLR

class Agent:
    def __init__(self, env_name, n_iter, n_states, action_bounds, n_actions, lr):
        self.env_name = env_name
        self.n_iter = n_iter
        self.action_bounds = action_bounds
        self.n_actions = n_actions
        self.n_states = n_states
        #self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        #self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        #print('Device:', torch.device("cuda" if torch.cuda.is_available else "cpu"))
        self.device = torch.device("cpu")
        self.lr = lr

        self.current_policy = Actor(n_states=self.n_states,
                                    n_actions=self.n_actions).to(self.device)
        self.critic = Critic(n_states=self.n_states).to(self.device)

        self.actor_optimizer = Adam(self.current_policy.parameters(), lr=self.lr, eps=1e-5)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)

        self.critic_loss = torch.nn.MSELoss()
        
        self.scheduler = lambda step: max(1.0 - float(step / self.n_iter), 0)
        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)
        self.critic_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)
        
        # 指数的減衰を使用したより洗練された学習率スケジューラーchatgpt
        """
        gamma_value = 0.95
        self.actor_scheduler = ExponentialLR(self.actor_optimizer, gamma=gamma_value)
        self.critic_scheduler = ExponentialLR(self.critic_optimizer, gamma=gamma_value)
        """

    def choose_dist(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            dist = self.current_policy(state)

        # action *= self.action_bounds[1]
        # action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

        return dist

    def get_value(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            value = self.critic(state)

        return value.detach().cpu().numpy()

    def optimize(self, actor_loss, critic_loss):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.current_policy.parameters(), 0.5)
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.current_policy.parameters(), 0.5)
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

    def schedule_lr(self):
        # self.total_scheduler.step()
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def save_weights(self, iteration, state_rms):
        torch.save({"current_policy_state_dict": self.current_policy.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                    "actor_scheduler_state_dict": self.actor_scheduler.state_dict(),
                    "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
                    "iteration": iteration,
                    "state_rms_mean": state_rms.mean,
                    "state_rms_var": state_rms.var,
                    "state_rms_count": state_rms.count}, "./journal/finetuning_lr6/finetuning_30per_10000itr_lr6/"+self.env_name + str(iteration)+ "_weights.pth")

    def load_weights(self):
        #checkpoint = torch.load("./cpuHumanoid/"+self.env_name + "100000_cuda_weights.pth")
        #checkpoint = torch.load("./pre-trained models/"+self.env_name + "_weights.pth") 
        checkpoint = torch.load("./models/Ant_advF_lr5_10_30000_weights.pth") 
        #checkpoint = torch.load("./journal/fine/"+self.env_name + "1000_weights.pth") 
        self.current_policy.load_state_dict(checkpoint["current_policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.actor_scheduler.load_state_dict(checkpoint["actor_scheduler_state_dict"])
        self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler_state_dict"])
        iteration = checkpoint["iteration"]
        state_rms_mean = checkpoint["state_rms_mean"]
        state_rms_var = checkpoint["state_rms_var"]

        return iteration, state_rms_mean, state_rms_var
    #finetuning 
    """
    def load_weights_f(self):
        checkpoint = torch.load("./journal/nomal/"+self.env_name + "100000_weights.pth") 
        self.current_policy.load_state_dict(checkpoint["current_policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        #self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        #self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        #self.actor_scheduler.load_state_dict(checkpoint["actor_scheduler_state_dict"])
        #self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler_state_dict"])
        #iteration = checkpoint["iteration"]
        #iteration = 0
        state_rms_mean = checkpoint["state_rms_mean"]
        state_rms_var = checkpoint["state_rms_var"]

        return state_rms_mean, state_rms_var
    """

    def set_to_eval_mode(self):
        self.current_policy.eval()
        self.critic.eval()

    def set_to_train_mode(self):
        self.current_policy.train()
        self.critic.train()
