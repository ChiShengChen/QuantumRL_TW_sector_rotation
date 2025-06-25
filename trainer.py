import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import Actor, Critic

class Memory:
    """A buffer for storing trajectories experienced by a PPO agent."""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

class PPOAgent:
    def __init__(self, model_type, model_config, ppo_config, device):
        self.device = device
        self.gamma = ppo_config['gamma']
        self.ppo_epochs = ppo_config['ppo_epochs']
        self.clip_eps = ppo_config['ppo_clip_eps']
        self.entropy_beta = ppo_config['entropy_beta']
        
        self.actor = Actor(model_type, model_config, ppo_config['output_dim']).to(device)
        self.critic = Critic(model_type, model_config).to(device)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=ppo_config['lr_actor'])
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=ppo_config['lr_critic'])
        
        self.mse_loss = nn.MSELoss()
        self.memory = Memory()

    def select_action(self, state, hidden_state_actor=None, hidden_state_critic=None):
        with torch.no_grad():
            # State needs to be shaped as (B, T, C) where B=1 for single inference
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            state_tensor = state.to(self.device).unsqueeze(0)
            
            dist, new_hidden_actor = self.actor(state_tensor, hidden_state_actor)
            value, new_hidden_critic = self.critic(state_tensor, hidden_state_critic)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Storing tensors that are already on the correct device
        self.memory.states.append(state_tensor)
        self.memory.actions.append(action)
        self.memory.logprobs.append(log_prob)
        self.memory.values.append(value)
        
        return action.item(), new_hidden_actor, new_hidden_critic

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.memory.states, dim=0), dim=1).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0), dim=1).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs, dim=0), dim=1).detach()
        old_values = torch.squeeze(torch.stack(self.memory.values, dim=0), dim=1).detach()

        # Generalized Advantage Estimation (GAE)
        advantages = rewards - old_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.ppo_epochs):
            # Evaluating old actions and values
            dist, _ = self.actor(old_states)
            values, _ = self.critic(old_states)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            logprobs = dist.log_prob(old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            
            # final loss of clipped objective PPO
            # The critic loss is often scaled by a coefficient (e.g., 0.5)
            # The entropy bonus encourages exploration
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * self.mse_loss(values, rewards.unsqueeze(1))
            entropy_bonus = dist.entropy().mean()
            
            loss = actor_loss + critic_loss - self.entropy_beta * entropy_bonus
            
            # take gradient step
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            # Optional: Gradient clipping for stability
            # nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            # nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.optimizer_actor.step()
            self.optimizer_critic.step()
            
        # Clear memory for the next trajectory collection
        self.memory.clear_memory() 