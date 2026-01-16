# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import HIMActorCritic
from rsl_rl.storage import HIMRolloutStorage

class HIMPPO:
    actor_critic: HIMActorCritic # 策略价值网络
    def __init__(self,
                 actor_critic,
                 use_flip = True,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 symmetry_scale=1e-3,
                 ):

        self.device = device
        self.use_flip = use_flip

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = HIMRolloutStorage.Transition()
        self.transition_sym = HIMRolloutStorage.Transition()
        self.symmetry_scale = symmetry_scale
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = HIMRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values，根据当前观测生成动作，同时记录相关信息
        # 这里调用了ActorCritic模块的act()，他自己内部再调用update_distribution()得到action_mean/std，存到transition
        self.transition.actions = self.actor_critic.act(obs).detach() # 得到动作
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach() # 得到value，评估状态价值
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach() # 计算动作对数概率
        self.transition.action_mean = self.actor_critic.action_mean.detach() # 动作均值
        self.transition.action_sigma = self.actor_critic.action_std.detach() # 动作标准差
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        
        obs_sym = self.flip_g1_actor_obs(obs)
        critic_obs_sym = self.flip_g1_critic_obs(critic_obs)
        self.transition_sym.actions = self.actor_critic.act(obs_sym).detach()
        self.transition_sym.values = self.actor_critic.evaluate(critic_obs_sym).detach()
        self.transition_sym.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition_sym.actions).detach()
        self.transition_sym.action_mean = self.actor_critic.action_mean.detach()
        self.transition_sym.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition_sym.observations = obs_sym
        self.transition_sym.critic_observations = critic_obs_sym
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        # 处理环境返回的结果
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        next_critic_obs_sym = self.flip_g1_critic_obs(next_critic_obs) # 对称化
        self.transition_sym.next_critic_observations = next_critic_obs_sym.clone()
        self.transition_sym.rewards = rewards.clone()
        self.transition_sym.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self.transition_sym.rewards += self.gamma * torch.squeeze(self.transition_sym.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.storage.add_transitions(self.transition_sym)
        self.transition.clear()
        self.transition_sym.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs): # 优势估计
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        mean_actor_sym_loss = 0
        mean_critic_sym_loss = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator:
                self.actor_critic.act(obs_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                #Estimator Update
                if self.use_flip: # 用来对称性
                    flipped_obs_batch = self.flip_g1_actor_obs(obs_batch)
                    flipped_next_critic_obs_batch = self.flip_g1_critic_obs(next_critic_obs_batch)
                    estimator_update_obs_batch =  torch.cat((obs_batch, flipped_obs_batch), dim=0)
                    estimator_update_next_critic_obs_batch = torch.cat((next_critic_obs_batch, flipped_next_critic_obs_batch), dim=0)
                else:
                    estimator_update_obs_batch = obs_batch
                    estimator_update_next_critic_obs_batch = next_critic_obs_batch
                # 额外增加actor，cirtic symmetry loss
                estimation_loss, swap_loss = self.actor_critic.update_estimator(estimator_update_obs_batch, estimator_update_next_critic_obs_batch, lr=self.learning_rate)
                
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                    
                if self.use_flip:
                    flipped_critic_obs_batch = self.flip_g1_critic_obs(critic_obs_batch)
                    actor_sym_loss = self.symmetry_scale * torch.mean(torch.sum(torch.square(self.actor_critic.act_inference(flipped_obs_batch) - self.flip_g1_actions(self.actor_critic.act_inference(obs_batch))), dim=-1))
                    critic_sym_loss = self.symmetry_scale * torch.mean(torch.square(self.actor_critic.evaluate(flipped_critic_obs_batch) - self.actor_critic.evaluate(critic_obs_batch).detach()))
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + actor_sym_loss + critic_sym_loss
                else:
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimation_loss += estimation_loss
                mean_swap_loss += swap_loss
                if self.use_flip:
                    mean_actor_sym_loss += actor_sym_loss.item()
                    mean_critic_sym_loss += critic_sym_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        if self.use_flip:
            mean_actor_sym_loss /= num_updates
            mean_critic_sym_loss /= num_updates
        self.storage.clear()

        if self.use_flip:
            return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss, mean_actor_sym_loss, mean_critic_sym_loss
        else:
            return mean_value_loss, mean_surrogate_loss, estimation_loss, swap_loss, 0, 0
    
    def flip_g1_actor_obs(self, obs):
        proprioceptive_obs = torch.clone(obs[:, :self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length])
        proprioceptive_obs = proprioceptive_obs.view(-1, self.actor_critic.actor_history_length, self.actor_critic.num_one_step_obs)
        
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3] # height command
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4] # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5] # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6] # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7] =  proprioceptive_obs[:, :, 7] # projected gravity x
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8] # projected gravity y
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9] # projected gravity z
        
                # ---- joint pos (53 dof) ----
        # legs
        flipped_proprioceptive_obs[:, :, 10 + 0] =  proprioceptive_obs[:, :, 10 + 6]
        flipped_proprioceptive_obs[:, :, 10 + 1] = -proprioceptive_obs[:, :, 10 + 7]
        flipped_proprioceptive_obs[:, :, 10 + 2] = -proprioceptive_obs[:, :, 10 + 8]
        flipped_proprioceptive_obs[:, :, 10 + 3] =  proprioceptive_obs[:, :, 10 + 9]
        flipped_proprioceptive_obs[:, :, 10 + 4] =  proprioceptive_obs[:, :, 10 + 10]
        flipped_proprioceptive_obs[:, :, 10 + 5] = -proprioceptive_obs[:, :, 10 + 11]

        flipped_proprioceptive_obs[:, :, 10 + 6]  =  proprioceptive_obs[:, :, 10 + 0]
        flipped_proprioceptive_obs[:, :, 10 + 7]  = -proprioceptive_obs[:, :, 10 + 1]
        flipped_proprioceptive_obs[:, :, 10 + 8]  = -proprioceptive_obs[:, :, 10 + 2]
        flipped_proprioceptive_obs[:, :, 10 + 9]  =  proprioceptive_obs[:, :, 10 + 3]
        flipped_proprioceptive_obs[:, :, 10 + 10] =  proprioceptive_obs[:, :, 10 + 4]
        flipped_proprioceptive_obs[:, :, 10 + 11] = -proprioceptive_obs[:, :, 10 + 5]

        # waist (yaw/roll flip sign, pitch keep)
        flipped_proprioceptive_obs[:, :, 10 + 12] = -proprioceptive_obs[:, :, 10 + 12]
        flipped_proprioceptive_obs[:, :, 10 + 13] = -proprioceptive_obs[:, :, 10 + 13]
        flipped_proprioceptive_obs[:, :, 10 + 14] =  proprioceptive_obs[:, :, 10 + 14]

        # arms
        flipped_proprioceptive_obs[:, :, 10 + 15] =  proprioceptive_obs[:, :, 10 + 34]
        flipped_proprioceptive_obs[:, :, 10 + 16] = -proprioceptive_obs[:, :, 10 + 35]
        flipped_proprioceptive_obs[:, :, 10 + 17] = -proprioceptive_obs[:, :, 10 + 36]
        flipped_proprioceptive_obs[:, :, 10 + 18] =  proprioceptive_obs[:, :, 10 + 37]
        flipped_proprioceptive_obs[:, :, 10 + 19] = -proprioceptive_obs[:, :, 10 + 38]
        flipped_proprioceptive_obs[:, :, 10 + 20] =  proprioceptive_obs[:, :, 10 + 39]
        flipped_proprioceptive_obs[:, :, 10 + 21] = -proprioceptive_obs[:, :, 10 + 40]

        flipped_proprioceptive_obs[:, :, 10 + 34] =  proprioceptive_obs[:, :, 10 + 15]
        flipped_proprioceptive_obs[:, :, 10 + 35] = -proprioceptive_obs[:, :, 10 + 16]
        flipped_proprioceptive_obs[:, :, 10 + 36] = -proprioceptive_obs[:, :, 10 + 17]
        flipped_proprioceptive_obs[:, :, 10 + 37] =  proprioceptive_obs[:, :, 10 + 18]
        flipped_proprioceptive_obs[:, :, 10 + 38] = -proprioceptive_obs[:, :, 10 + 19]
        flipped_proprioceptive_obs[:, :, 10 + 39] =  proprioceptive_obs[:, :, 10 + 20]
        flipped_proprioceptive_obs[:, :, 10 + 40] = -proprioceptive_obs[:, :, 10 + 21]

        # inspire hands (thumb yaw flips sign; others keep)
        flipped_proprioceptive_obs[:, :, 10 + 22] = -proprioceptive_obs[:, :, 10 + 41]
        flipped_proprioceptive_obs[:, :, 10 + 23] =  proprioceptive_obs[:, :, 10 + 42]
        flipped_proprioceptive_obs[:, :, 10 + 24] =  proprioceptive_obs[:, :, 10 + 43]
        flipped_proprioceptive_obs[:, :, 10 + 25] =  proprioceptive_obs[:, :, 10 + 44]
        flipped_proprioceptive_obs[:, :, 10 + 26] =  proprioceptive_obs[:, :, 10 + 45]
        flipped_proprioceptive_obs[:, :, 10 + 27] =  proprioceptive_obs[:, :, 10 + 46]
        flipped_proprioceptive_obs[:, :, 10 + 28] =  proprioceptive_obs[:, :, 10 + 47]
        flipped_proprioceptive_obs[:, :, 10 + 29] =  proprioceptive_obs[:, :, 10 + 48]
        flipped_proprioceptive_obs[:, :, 10 + 30] =  proprioceptive_obs[:, :, 10 + 49]
        flipped_proprioceptive_obs[:, :, 10 + 31] =  proprioceptive_obs[:, :, 10 + 50]
        flipped_proprioceptive_obs[:, :, 10 + 32] =  proprioceptive_obs[:, :, 10 + 51]
        flipped_proprioceptive_obs[:, :, 10 + 33] =  proprioceptive_obs[:, :, 10 + 52]

        flipped_proprioceptive_obs[:, :, 10 + 41] = -proprioceptive_obs[:, :, 10 + 22]
        flipped_proprioceptive_obs[:, :, 10 + 42] =  proprioceptive_obs[:, :, 10 + 23]
        flipped_proprioceptive_obs[:, :, 10 + 43] =  proprioceptive_obs[:, :, 10 + 24]
        flipped_proprioceptive_obs[:, :, 10 + 44] =  proprioceptive_obs[:, :, 10 + 25]
        flipped_proprioceptive_obs[:, :, 10 + 45] =  proprioceptive_obs[:, :, 10 + 26]
        flipped_proprioceptive_obs[:, :, 10 + 46] =  proprioceptive_obs[:, :, 10 + 27]
        flipped_proprioceptive_obs[:, :, 10 + 47] =  proprioceptive_obs[:, :, 10 + 28]
        flipped_proprioceptive_obs[:, :, 10 + 48] =  proprioceptive_obs[:, :, 10 + 29]
        flipped_proprioceptive_obs[:, :, 10 + 49] =  proprioceptive_obs[:, :, 10 + 30]
        flipped_proprioceptive_obs[:, :, 10 + 50] =  proprioceptive_obs[:, :, 10 + 31]
        flipped_proprioceptive_obs[:, :, 10 + 51] =  proprioceptive_obs[:, :, 10 + 32]
        flipped_proprioceptive_obs[:, :, 10 + 52] =  proprioceptive_obs[:, :, 10 + 33]


        # ---- joint vel (offset +53) ----
        # legs
        flipped_proprioceptive_obs[:, :, 10 + 53 + 0] =  proprioceptive_obs[:, :, 10 + 53 + 6]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 1] = -proprioceptive_obs[:, :, 10 + 53 + 7]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 2] = -proprioceptive_obs[:, :, 10 + 53 + 8]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 3] =  proprioceptive_obs[:, :, 10 + 53 + 9]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 4] =  proprioceptive_obs[:, :, 10 + 53 + 10]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 5] = -proprioceptive_obs[:, :, 10 + 53 + 11]

        flipped_proprioceptive_obs[:, :, 10 + 53 + 6]  =  proprioceptive_obs[:, :, 10 + 53 + 0]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 7]  = -proprioceptive_obs[:, :, 10 + 53 + 1]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 8]  = -proprioceptive_obs[:, :, 10 + 53 + 2]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 9]  =  proprioceptive_obs[:, :, 10 + 53 + 3]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 10] =  proprioceptive_obs[:, :, 10 + 53 + 4]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 11] = -proprioceptive_obs[:, :, 10 + 53 + 5]

        # waist
        flipped_proprioceptive_obs[:, :, 10 + 53 + 12] = -proprioceptive_obs[:, :, 10 + 53 + 12]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 13] = -proprioceptive_obs[:, :, 10 + 53 + 13]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 14] =  proprioceptive_obs[:, :, 10 + 53 + 14]

        # arms
        flipped_proprioceptive_obs[:, :, 10 + 53 + 15] =  proprioceptive_obs[:, :, 10 + 53 + 34]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 16] = -proprioceptive_obs[:, :, 10 + 53 + 35]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 17] = -proprioceptive_obs[:, :, 10 + 53 + 36]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 18] =  proprioceptive_obs[:, :, 10 + 53 + 37]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 19] = -proprioceptive_obs[:, :, 10 + 53 + 38]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 20] =  proprioceptive_obs[:, :, 10 + 53 + 39]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 21] = -proprioceptive_obs[:, :, 10 + 53 + 40]

        flipped_proprioceptive_obs[:, :, 10 + 53 + 34] =  proprioceptive_obs[:, :, 10 + 53 + 15]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 35] = -proprioceptive_obs[:, :, 10 + 53 + 16]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 36] = -proprioceptive_obs[:, :, 10 + 53 + 17]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 37] =  proprioceptive_obs[:, :, 10 + 53 + 18]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 38] = -proprioceptive_obs[:, :, 10 + 53 + 19]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 39] =  proprioceptive_obs[:, :, 10 + 53 + 20]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 40] = -proprioceptive_obs[:, :, 10 + 53 + 21]

        # inspire hands
        flipped_proprioceptive_obs[:, :, 10 + 53 + 22] = -proprioceptive_obs[:, :, 10 + 53 + 41]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 23] =  proprioceptive_obs[:, :, 10 + 53 + 42]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 24] =  proprioceptive_obs[:, :, 10 + 53 + 43]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 25] =  proprioceptive_obs[:, :, 10 + 53 + 44]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 26] =  proprioceptive_obs[:, :, 10 + 53 + 45]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 27] =  proprioceptive_obs[:, :, 10 + 53 + 46]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 28] =  proprioceptive_obs[:, :, 10 + 53 + 47]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 29] =  proprioceptive_obs[:, :, 10 + 53 + 48]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 30] =  proprioceptive_obs[:, :, 10 + 53 + 49]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 31] =  proprioceptive_obs[:, :, 10 + 53 + 50]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 32] =  proprioceptive_obs[:, :, 10 + 53 + 51]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 33] =  proprioceptive_obs[:, :, 10 + 53 + 52]

        flipped_proprioceptive_obs[:, :, 10 + 53 + 41] = -proprioceptive_obs[:, :, 10 + 53 + 22]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 42] =  proprioceptive_obs[:, :, 10 + 53 + 23]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 43] =  proprioceptive_obs[:, :, 10 + 53 + 24]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 44] =  proprioceptive_obs[:, :, 10 + 53 + 25]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 45] =  proprioceptive_obs[:, :, 10 + 53 + 26]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 46] =  proprioceptive_obs[:, :, 10 + 53 + 27]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 47] =  proprioceptive_obs[:, :, 10 + 53 + 28]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 48] =  proprioceptive_obs[:, :, 10 + 53 + 29]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 49] =  proprioceptive_obs[:, :, 10 + 53 + 30]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 50] =  proprioceptive_obs[:, :, 10 + 53 + 31]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 51] =  proprioceptive_obs[:, :, 10 + 53 + 32]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 52] =  proprioceptive_obs[:, :, 10 + 53 + 33]


                # ---- joint target (actions, offset +106) ----
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 0] =  proprioceptive_obs[:, :, 10 + 2 * 53 + 6]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 1] = -proprioceptive_obs[:, :, 10 + 2 * 53 + 7]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 2] = -proprioceptive_obs[:, :, 10 + 2 * 53 + 8]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 3] =  proprioceptive_obs[:, :, 10 + 2 * 53 + 9]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 4] =  proprioceptive_obs[:, :, 10 + 2 * 53 + 10]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 5] = -proprioceptive_obs[:, :, 10 + 2 * 53 + 11]

        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 6]  =  proprioceptive_obs[:, :, 10 + 2 * 53 + 0]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 7]  = -proprioceptive_obs[:, :, 10 + 2 * 53 + 1]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 8]  = -proprioceptive_obs[:, :, 10 + 2 * 53 + 2]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 9]  =  proprioceptive_obs[:, :, 10 + 2 * 53 + 3]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 10] =  proprioceptive_obs[:, :, 10 + 2 * 53 + 4]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 11] = -proprioceptive_obs[:, :, 10 + 2 * 53 + 5]


        return flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length).detach()                                                                                                                                                                                                                                             
    
    def flip_g1_critic_obs(self, critic_obs):
        proprioceptive_obs = torch.clone(critic_obs[:, :self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length])
        proprioceptive_obs = proprioceptive_obs.view(-1, self.actor_critic.critic_history_length, self.actor_critic.num_one_step_critic_obs)
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3] # height command
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4] # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5] # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6] # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7] =  proprioceptive_obs[:, :, 7] # projected gravity x
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8] # projected gravity y
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9] # projected gravity z
        
        # ---- joint pos (53 dof) ----
        # legs
        flipped_proprioceptive_obs[:, :, 10 + 0] =  proprioceptive_obs[:, :, 10 + 6]
        flipped_proprioceptive_obs[:, :, 10 + 1] = -proprioceptive_obs[:, :, 10 + 7]
        flipped_proprioceptive_obs[:, :, 10 + 2] = -proprioceptive_obs[:, :, 10 + 8]
        flipped_proprioceptive_obs[:, :, 10 + 3] =  proprioceptive_obs[:, :, 10 + 9]
        flipped_proprioceptive_obs[:, :, 10 + 4] =  proprioceptive_obs[:, :, 10 + 10]
        flipped_proprioceptive_obs[:, :, 10 + 5] = -proprioceptive_obs[:, :, 10 + 11]

        flipped_proprioceptive_obs[:, :, 10 + 6]  =  proprioceptive_obs[:, :, 10 + 0]
        flipped_proprioceptive_obs[:, :, 10 + 7]  = -proprioceptive_obs[:, :, 10 + 1]
        flipped_proprioceptive_obs[:, :, 10 + 8]  = -proprioceptive_obs[:, :, 10 + 2]
        flipped_proprioceptive_obs[:, :, 10 + 9]  =  proprioceptive_obs[:, :, 10 + 3]
        flipped_proprioceptive_obs[:, :, 10 + 10] =  proprioceptive_obs[:, :, 10 + 4]
        flipped_proprioceptive_obs[:, :, 10 + 11] = -proprioceptive_obs[:, :, 10 + 5]

        # waist (yaw/roll flip sign, pitch keep)
        flipped_proprioceptive_obs[:, :, 10 + 12] = -proprioceptive_obs[:, :, 10 + 12]
        flipped_proprioceptive_obs[:, :, 10 + 13] = -proprioceptive_obs[:, :, 10 + 13]
        flipped_proprioceptive_obs[:, :, 10 + 14] =  proprioceptive_obs[:, :, 10 + 14]

        # arms
        flipped_proprioceptive_obs[:, :, 10 + 15] =  proprioceptive_obs[:, :, 10 + 34]
        flipped_proprioceptive_obs[:, :, 10 + 16] = -proprioceptive_obs[:, :, 10 + 35]
        flipped_proprioceptive_obs[:, :, 10 + 17] = -proprioceptive_obs[:, :, 10 + 36]
        flipped_proprioceptive_obs[:, :, 10 + 18] =  proprioceptive_obs[:, :, 10 + 37]
        flipped_proprioceptive_obs[:, :, 10 + 19] = -proprioceptive_obs[:, :, 10 + 38]
        flipped_proprioceptive_obs[:, :, 10 + 20] =  proprioceptive_obs[:, :, 10 + 39]
        flipped_proprioceptive_obs[:, :, 10 + 21] = -proprioceptive_obs[:, :, 10 + 40]

        flipped_proprioceptive_obs[:, :, 10 + 34] =  proprioceptive_obs[:, :, 10 + 15]
        flipped_proprioceptive_obs[:, :, 10 + 35] = -proprioceptive_obs[:, :, 10 + 16]
        flipped_proprioceptive_obs[:, :, 10 + 36] = -proprioceptive_obs[:, :, 10 + 17]
        flipped_proprioceptive_obs[:, :, 10 + 37] =  proprioceptive_obs[:, :, 10 + 18]
        flipped_proprioceptive_obs[:, :, 10 + 38] = -proprioceptive_obs[:, :, 10 + 19]
        flipped_proprioceptive_obs[:, :, 10 + 39] =  proprioceptive_obs[:, :, 10 + 20]
        flipped_proprioceptive_obs[:, :, 10 + 40] = -proprioceptive_obs[:, :, 10 + 21]

        # inspire hands (thumb yaw flips sign; others keep)
        flipped_proprioceptive_obs[:, :, 10 + 22] = -proprioceptive_obs[:, :, 10 + 41]
        flipped_proprioceptive_obs[:, :, 10 + 23] =  proprioceptive_obs[:, :, 10 + 42]
        flipped_proprioceptive_obs[:, :, 10 + 24] =  proprioceptive_obs[:, :, 10 + 43]
        flipped_proprioceptive_obs[:, :, 10 + 25] =  proprioceptive_obs[:, :, 10 + 44]
        flipped_proprioceptive_obs[:, :, 10 + 26] =  proprioceptive_obs[:, :, 10 + 45]
        flipped_proprioceptive_obs[:, :, 10 + 27] =  proprioceptive_obs[:, :, 10 + 46]
        flipped_proprioceptive_obs[:, :, 10 + 28] =  proprioceptive_obs[:, :, 10 + 47]
        flipped_proprioceptive_obs[:, :, 10 + 29] =  proprioceptive_obs[:, :, 10 + 48]
        flipped_proprioceptive_obs[:, :, 10 + 30] =  proprioceptive_obs[:, :, 10 + 49]
        flipped_proprioceptive_obs[:, :, 10 + 31] =  proprioceptive_obs[:, :, 10 + 50]
        flipped_proprioceptive_obs[:, :, 10 + 32] =  proprioceptive_obs[:, :, 10 + 51]
        flipped_proprioceptive_obs[:, :, 10 + 33] =  proprioceptive_obs[:, :, 10 + 52]

        flipped_proprioceptive_obs[:, :, 10 + 41] = -proprioceptive_obs[:, :, 10 + 22]
        flipped_proprioceptive_obs[:, :, 10 + 42] =  proprioceptive_obs[:, :, 10 + 23]
        flipped_proprioceptive_obs[:, :, 10 + 43] =  proprioceptive_obs[:, :, 10 + 24]
        flipped_proprioceptive_obs[:, :, 10 + 44] =  proprioceptive_obs[:, :, 10 + 25]
        flipped_proprioceptive_obs[:, :, 10 + 45] =  proprioceptive_obs[:, :, 10 + 26]
        flipped_proprioceptive_obs[:, :, 10 + 46] =  proprioceptive_obs[:, :, 10 + 27]
        flipped_proprioceptive_obs[:, :, 10 + 47] =  proprioceptive_obs[:, :, 10 + 28]
        flipped_proprioceptive_obs[:, :, 10 + 48] =  proprioceptive_obs[:, :, 10 + 29]
        flipped_proprioceptive_obs[:, :, 10 + 49] =  proprioceptive_obs[:, :, 10 + 30]
        flipped_proprioceptive_obs[:, :, 10 + 50] =  proprioceptive_obs[:, :, 10 + 31]
        flipped_proprioceptive_obs[:, :, 10 + 51] =  proprioceptive_obs[:, :, 10 + 32]
        flipped_proprioceptive_obs[:, :, 10 + 52] =  proprioceptive_obs[:, :, 10 + 33]


        # ---- joint vel (offset +53) ----
        # legs
        flipped_proprioceptive_obs[:, :, 10 + 53 + 0] =  proprioceptive_obs[:, :, 10 + 53 + 6]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 1] = -proprioceptive_obs[:, :, 10 + 53 + 7]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 2] = -proprioceptive_obs[:, :, 10 + 53 + 8]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 3] =  proprioceptive_obs[:, :, 10 + 53 + 9]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 4] =  proprioceptive_obs[:, :, 10 + 53 + 10]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 5] = -proprioceptive_obs[:, :, 10 + 53 + 11]

        flipped_proprioceptive_obs[:, :, 10 + 53 + 6]  =  proprioceptive_obs[:, :, 10 + 53 + 0]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 7]  = -proprioceptive_obs[:, :, 10 + 53 + 1]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 8]  = -proprioceptive_obs[:, :, 10 + 53 + 2]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 9]  =  proprioceptive_obs[:, :, 10 + 53 + 3]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 10] =  proprioceptive_obs[:, :, 10 + 53 + 4]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 11] = -proprioceptive_obs[:, :, 10 + 53 + 5]

        # waist
        flipped_proprioceptive_obs[:, :, 10 + 53 + 12] = -proprioceptive_obs[:, :, 10 + 53 + 12]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 13] = -proprioceptive_obs[:, :, 10 + 53 + 13]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 14] =  proprioceptive_obs[:, :, 10 + 53 + 14]

        # arms
        flipped_proprioceptive_obs[:, :, 10 + 53 + 15] =  proprioceptive_obs[:, :, 10 + 53 + 34]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 16] = -proprioceptive_obs[:, :, 10 + 53 + 35]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 17] = -proprioceptive_obs[:, :, 10 + 53 + 36]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 18] =  proprioceptive_obs[:, :, 10 + 53 + 37]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 19] = -proprioceptive_obs[:, :, 10 + 53 + 38]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 20] =  proprioceptive_obs[:, :, 10 + 53 + 39]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 21] = -proprioceptive_obs[:, :, 10 + 53 + 40]

        flipped_proprioceptive_obs[:, :, 10 + 53 + 34] =  proprioceptive_obs[:, :, 10 + 53 + 15]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 35] = -proprioceptive_obs[:, :, 10 + 53 + 16]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 36] = -proprioceptive_obs[:, :, 10 + 53 + 17]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 37] =  proprioceptive_obs[:, :, 10 + 53 + 18]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 38] = -proprioceptive_obs[:, :, 10 + 53 + 19]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 39] =  proprioceptive_obs[:, :, 10 + 53 + 20]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 40] = -proprioceptive_obs[:, :, 10 + 53 + 21]

        # inspire hands
        flipped_proprioceptive_obs[:, :, 10 + 53 + 22] = -proprioceptive_obs[:, :, 10 + 53 + 41]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 23] =  proprioceptive_obs[:, :, 10 + 53 + 42]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 24] =  proprioceptive_obs[:, :, 10 + 53 + 43]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 25] =  proprioceptive_obs[:, :, 10 + 53 + 44]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 26] =  proprioceptive_obs[:, :, 10 + 53 + 45]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 27] =  proprioceptive_obs[:, :, 10 + 53 + 46]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 28] =  proprioceptive_obs[:, :, 10 + 53 + 47]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 29] =  proprioceptive_obs[:, :, 10 + 53 + 48]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 30] =  proprioceptive_obs[:, :, 10 + 53 + 49]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 31] =  proprioceptive_obs[:, :, 10 + 53 + 50]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 32] =  proprioceptive_obs[:, :, 10 + 53 + 51]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 33] =  proprioceptive_obs[:, :, 10 + 53 + 52]

        flipped_proprioceptive_obs[:, :, 10 + 53 + 41] = -proprioceptive_obs[:, :, 10 + 53 + 22]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 42] =  proprioceptive_obs[:, :, 10 + 53 + 23]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 43] =  proprioceptive_obs[:, :, 10 + 53 + 24]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 44] =  proprioceptive_obs[:, :, 10 + 53 + 25]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 45] =  proprioceptive_obs[:, :, 10 + 53 + 26]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 46] =  proprioceptive_obs[:, :, 10 + 53 + 27]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 47] =  proprioceptive_obs[:, :, 10 + 53 + 28]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 48] =  proprioceptive_obs[:, :, 10 + 53 + 29]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 49] =  proprioceptive_obs[:, :, 10 + 53 + 30]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 50] =  proprioceptive_obs[:, :, 10 + 53 + 31]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 51] =  proprioceptive_obs[:, :, 10 + 53 + 32]
        flipped_proprioceptive_obs[:, :, 10 + 53 + 52] =  proprioceptive_obs[:, :, 10 + 53 + 33]


        # ---- joint target (actions, offset +106) ----
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 0] =  proprioceptive_obs[:, :, 10 + 2 * 53 + 6]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 1] = -proprioceptive_obs[:, :, 10 + 2 * 53 + 7]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 2] = -proprioceptive_obs[:, :, 10 + 2 * 53 + 8]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 3] =  proprioceptive_obs[:, :, 10 + 2 * 53 + 9]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 4] =  proprioceptive_obs[:, :, 10 + 2 * 53 + 10]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 5] = -proprioceptive_obs[:, :, 10 + 2 * 53 + 11]

        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 6]  =  proprioceptive_obs[:, :, 10 + 2 * 53 + 0]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 7]  = -proprioceptive_obs[:, :, 10 + 2 * 53 + 1]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 8]  = -proprioceptive_obs[:, :, 10 + 2 * 53 + 2]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 9]  =  proprioceptive_obs[:, :, 10 + 2 * 53 + 3]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 10] =  proprioceptive_obs[:, :, 10 + 2 * 53 + 4]
        flipped_proprioceptive_obs[:, :, 10 + 2 * 53 + 11] = -proprioceptive_obs[:, :, 10 + 2 * 53 + 5]


        return flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length).detach()
    
    def flip_g1_actions(self, actions):
        flipped_actions = torch.zeros_like(actions)
        flipped_actions[:,  0] =  actions[:, 6]        # 0 "left_hip_pitch_joint",
        flipped_actions[:,  1] = -actions[:, 7]        # 1 "left_hip_roll_joint",
        flipped_actions[:,  2] = -actions[:, 8]        # 2 "left_hip_yaw_joint",
        flipped_actions[:,  3] =  actions[:, 9]        # 3 "left_knee_joint",
        flipped_actions[:,  4] =  actions[:, 10]       # 4 "left_ankle_pitch_joint",
        flipped_actions[:,  5] = -actions[:, 11]       # 5 "left_ankle_roll_joint",
        flipped_actions[:,  6] =  actions[:, 0]        # 6 "right_hip_pitch_joint",
        flipped_actions[:,  7] = -actions[:, 1]        # 7 "right_hip_roll_joint",
        flipped_actions[:,  8] = -actions[:, 2]        # 8 "right_hip_yaw_joint",
        flipped_actions[:,  9] =  actions[:, 3]        # 9 "right_knee_joint",
        flipped_actions[:, 10] =  actions[:, 4]        # 10 "right_ankle_pitch_joint",
        flipped_actions[:, 11] = -actions[:, 5]        # 11 "right_ankle_roll_joint",
        return flipped_actions.detach()