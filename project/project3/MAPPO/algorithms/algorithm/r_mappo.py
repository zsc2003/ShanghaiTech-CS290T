import numpy as np
import torch
import torch.nn as nn
from utils.util import huber_loss
from utils.valuenorm import ValueNorm
from algorithms.utils.util import check


class RMAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self.value_normalizer = ValueNorm(1, device=self.device)

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.

        :return value_loss: (torch.Tensor) value function loss.
        """

        # Calculate the clipped value predictions by adjusting the old predictions with the current predictions.
        # Ensuring the adjustment is within the specified clipping range.
        value_pred_clipped = values + torch.clamp(value_preds_batch - self.clip_param, value_preds_batch + self.clip_param)


        # Update the value normalizer with the current returns to ensure proper normalization for loss calculation.
        # You may use 'self.value_normalizer' here.
        self.value_normalizer.update(return_batch)
        return_batch_normalized = self.value_normalizer.normalize(return_batch)


        # Compute the errors for both the clipped and original predictions by normalizing the returns \
        # and subtracting the respective value predictions.
        error_clipped = return_batch_normalized - value_pred_clipped
        error_original = return_batch_normalized - values


        # Calculate the value losses using *Huber loss* for both the clipped and original errors.
        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)


        # The final value loss is the maximum of the clipped and original losses, averaged over the batch.
        value_loss = torch.max(value_loss_original, value_loss_clipped).mean()


        return value_loss


    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, _, old_action_log_probs_batch, \
        adv_targ, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                            obs_batch,
                                                                            rnn_states_batch,
                                                                            rnn_states_critic_batch,
                                                                            actions_batch,
                                                                            masks_batch)
        # Actor Update
        # You may use 'self.policy.actor_optimizer' to update the actor network.
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        policy_loss_origin = adv_targ * imp_weights
        policy_loss_clipped = adv_targ * torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
        policy_loss = torch.min(policy_loss_origin, policy_loss_clipped).mean()
        #  use gradient **ascent** -> negative for gradient **descent**
        actor_loss = -policy_loss - dist_entropy * self.entropy_coef

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            actor_loss.backward()
            self.policy.actor_optimizer.step()
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = 0

        # Critic Update
        # You may use 'self.policy.critic_optimizer' to update the critic network.
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)
        value_loss = value_loss * self.value_loss_coef

        self.policy.critic_optimizer.zero_grad()
        value_loss.backward()
        self.policy.critic_optimizer.step()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)


        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()