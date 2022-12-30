import numpy as np
import torch
from safepo.algos.policy_gradient import PG
from safepo.common.core import discount_cumsum
# from safepo.common.core import ConstrainedPolicyGradientAlgorithm
from safepo.algos.lagrangian_base import Lagrangian
import safepo.common.mpi_tools as mpi_tools


class CRPO(PG):
    def __init__(
            self,
            algo='crpo',
            cost_limit=25.,
            use_cost_value_function=True,
            **kwargs
    ):
        PG.__init__(
            self,
            algo=algo,
            use_cost_value_function=use_cost_value_function,
            **kwargs
        )
        self.cost_limit = cost_limit

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()

    def compute_loss_pi(self, data: dict):
        '''
            computing pi/actor loss

            Returns:
                torch.Tensor
        '''
        ep_costs = self.logger.get_stats('EpCosts')[0]
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        if ep_costs > self.cost_limit:
            # Compute loss via ratio and advantage
            loss_pi = (ratio * data['cost_adv']).mean()
            loss_pi += self.entropy_coef * dist.entropy().mean()
        else:
            # Compute loss via ratio and advantage
            loss_pi = -(ratio * data['adv']).mean()
            loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()

        # Compute policy's entropy
        ent = dist.entropy().mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info