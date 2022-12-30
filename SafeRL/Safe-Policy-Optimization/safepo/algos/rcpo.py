import numpy as np
import torch
from safepo.algos.policy_gradient import PG
from safepo.common.core import discount_cumsum
# from safepo.common.core import ConstrainedPolicyGradientAlgorithm
from safepo.algos.lagrangian_base import Lagrangian
import safepo.common.mpi_tools as mpi_tools


class RCPO(PG, Lagrangian):
    def __init__(
            self,
            algo='rcpo',
            cost_limit=25.,
            lagrangian_multiplier_init=0.,
            vf_lr=0.0015,
            pi_lr=0.001,
            lambda_lr=0.0005,
            lambda_optimizer='Adam',
            use_cost_value_function=False,
            **kwargs
    ):
        PG.__init__(
            self,
            algo=algo,
            vf_lr=vf_lr,
            pi_lr=pi_lr,
            use_cost_value_function=use_cost_value_function,
            **kwargs
        )

        Lagrangian.__init__(
            self,
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer
        )

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('LagrangeMultiplier',
                                self.lagrangian_multiplier.item())

    def compute_loss_pi(self, data: dict):
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        rewards = []
        R = 0
        lagrangian_multiplier = self.lagrangian_multiplier
        for i in range(len(data['rew']) - 1, -1, -1):
            r = data['rew'][i]
            c = data['cost_adv'][i]
            R = r - lagrangian_multiplier * c + self.gamma * R
            rewards.insert(0, R)
        V = self.ac.v(data['obs']).detach()
        rewards = torch.tensor(rewards)
        loss_pi = -(_log_p * (rewards - V)).sum()

        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()

        # Compute policy's entropy
        ent = dist.entropy().mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats('EpCosts')[0]
        # First update policy and value network
        self.update_policy_net(data=data)
        self.update_value_net(data=data)
        # Now update Lagrange multiplier parameter
        self.update_lagrange_multiplier(ep_costs)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)

    def update_policy_net(self, data) -> None:
        # Get prob. distribution before updates: used to measure KL distance
        with torch.no_grad():
            self.p_dist = self.ac.pi.detach_dist(data['obs'])

        # Get loss and info values before update
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        self.loss_pi_before = pi_l_old.item()
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        loss_pi.backward()
        mpi_tools.mpi_avg_grads(self.ac.pi.net)
        self.pi_optimizer.step()

        q_dist = self.ac.pi.dist(data['obs'])
        torch_kl = torch.distributions.kl.kl_divergence(
            self.p_dist, q_dist).mean().item()

        # Track when policy iteration is stopped; Log changes from update
        self.logger.store(**{
            'Loss/Pi': self.loss_pi_before,
            'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
            'Misc/StopIter': 1,
            'Values/Adv': data['adv'].numpy(),
            'Entropy': pi_info['ent'],
            'KL': torch_kl,
            'PolicyRatio': pi_info['ratio']
        })

    def update_value_net(self, data: dict) -> None:
        rewards = []
        R = 0
        lagrangian_multiplier = self.lagrangian_multiplier
        for i in range(len(data['rew']) - 1, -1, -1):
            r = data['rew'][i]
            c = data['cost_adv'][i]
            R = r - lagrangian_multiplier * c + self.gamma + R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        loss_v = self.compute_loss_v(data['obs'], rewards)
        self.loss_v_before = loss_v.item()

        loss_v.backward()
        val_losses = loss_v.item()
        mpi_tools.mpi_avg_grads(self.ac.v)
        self.vf_optimizer.step()

        self.logger.store(**{
            'Loss/DeltaValue': np.mean(val_losses) - self.loss_v_before,
            'Loss/Value': self.loss_v_before,
        })
