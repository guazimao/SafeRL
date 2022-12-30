import numpy as np
from torch import optim
import torch
from safepo.algos.policy_gradient import PG
from safepo.algos.lagrangian_base import Lagrangian

class PDO(PG, Lagrangian):
    def __init__(self,
                 algo="pdo",
                 cost_limit=25.,
                 lagrangian_multiplier_init=0.001,
                 lambda_lr=0.032,
                 lambda_optimizer='Adam',
                 use_cost_value_function=True,
                 **kwargs):
        PG.__init__(
            self,
            algo=algo,
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

    def compute_loss_pi(self, data: dict):
        '''
            computing pi/actor loss

            Returns:
                torch.Tensor
        '''
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        # Compute loss via ratio and advantage
        lagrangian_multiplier = self.lambda_range_projection(self.lagrangian_multiplier).item()
        adv = data['adv'] - lagrangian_multiplier * data['cost_adv']
        loss_pi = -(ratio * adv).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()

        # Compute policy's entropy
        ent = dist.entropy().mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('LagrangeMultiplier',
                                self.lagrangian_multiplier.item())

    def update(self):
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # sub-sampling accelerates calculations
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats('EpCosts')[0]
        # First update Lagrange multiplier parameter
        self.update_lagrange_multiplier(ep_costs)
        # now update policy and value network
        self.update_policy_net(data=data)
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)