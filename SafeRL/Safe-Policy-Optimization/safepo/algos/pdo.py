import numpy as np
from torch import optim
import torch
from safepo.algos.policy_gradient import PG
from safepo.algos.cpo import CPO
from safepo.algos.lagrangian_base import Lagrangian
import safepo.common.mpi_tools as mpi_tools
from safepo.common.utils import get_flat_params_from, set_param_values_to_model,\
                                set_param_values_to_model,get_flat_gradients_from,\
                                conjugate_gradients

class PDO(CPO, Lagrangian):
    def __init__(self,
                 algo="pdo",
                 cost_limit=25.,
                 lagrangian_multiplier_init=0.001,
                 lambda_lr=0.035,
                 lambda_optimizer='Adam',
                 use_cost_value_function=True,
                 **kwargs):
        CPO.__init__(
            self,
            algo=algo,
            **kwargs
        )
        Lagrangian.__init__(
            self,
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer
        )

    def compute_loss_cost_performance(self, data):
        """
           Performance of cost on this moment
        """
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        cost_loss = (ratio * data['cost_adv']).mean()
        # ent = dist.entropy().mean().item()
        info = {}
        return cost_loss, info

    def adjust_step_direction(self,
                              step_dir,
                              g_flat,
                              p_dist,
                              data,
                              total_steps: int = 15,
                              decay: float = 0.8
                              ) -> tuple:
        """ TRPO performs line-search until constraint satisfaction."""
        step_frac = 1.0
        _theta_old = get_flat_params_from(self.ac.pi.net)
        expected_improve = g_flat.dot(step_dir)

        # while not within_trust_region:
        for j in range(total_steps):
            new_theta = _theta_old + step_frac * step_dir
            set_param_values_to_model(self.ac.pi.net, new_theta)
            acceptance_step = j + 1

            with torch.no_grad():
                loss_pi, pi_info = self.compute_loss_pi(data=data)
                # determine KL div between new and old policy
                q_dist = self.ac.pi.dist(data['obs'])
                torch_kl = torch.distributions.kl.kl_divergence(
                    p_dist, q_dist).mean().item()
            loss_improve = self.loss_pi_before - loss_pi.item()
            # average processes....
            torch_kl = mpi_tools.mpi_avg(torch_kl)
            loss_improve = mpi_tools.mpi_avg(loss_improve)

            self.logger.log("Expected Improvement: %.3f Actual: %.3f" % (
                expected_improve, loss_improve))
            if not torch.isfinite(loss_pi):
                self.logger.log('WARNING: loss_pi not finite')
            elif loss_improve < 0:
                self.logger.log('INFO: did not improve improve <0')
            elif torch_kl > self.target_kl * 1.5:
                self.logger.log('INFO: violated KL constraint.')
            else:
                # step only if surrogate is improved and when within trust reg.
                self.logger.log(f'Accept step at i={acceptance_step}')
                break
            step_frac *= decay
        else:
            self.logger.log('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        set_param_values_to_model(self.ac.pi.net, _theta_old)

        return step_frac * step_dir, acceptance_step

    def algorithm_specific_logs(self):
        self.logger.log_tabular('Misc/AcceptanceStep')
        self.logger.log_tabular('Misc/Alpha')
        self.logger.log_tabular('Misc/FinalStepNorm')
        self.logger.log_tabular('Misc/gradient_norm')
        self.logger.log_tabular('Misc/xHx')
        self.logger.log_tabular('Misc/H_inv_g')
        self.logger.log_tabular('Misc/cost_gradient_norm')
        self.logger.log_tabular('LagrangeMultiplier',
                                self.lagrangian_multiplier.item())

    def update(self):
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # sub-sampling accelerates calculations
        self.fvp_obs = data['obs'][::4]
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

    def update_policy_net(self, data):
        # Get loss and info values before update
        theta_old = get_flat_params_from(self.ac.pi.net)
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        self.loss_pi_before = loss_pi.item()
        self.loss_v_before = self.compute_loss_v(data['obs'],
                                                 data['target_v']).item()
        self.loss_c_before = self.compute_loss_c(data['obs'],
                                                 data['target_c']).item()
        # get prob. distribution before updates
        p_dist = self.ac.pi.dist(data['obs'])
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        # average grads across MPI processes
        mpi_tools.mpi_avg_grads(self.ac.pi.net)
        g_flat = get_flat_gradients_from(self.ac.pi.net)
        g_flat *= -1  # flip sign since policy_loss = -(ration * adv)

        # get the policy cost performance gradient b (flat as vector)
        self.pi_optimizer.zero_grad()
        loss_cost, _ = self.compute_loss_cost_performance(data=data)
        loss_cost.backward()
        # average grads across MPI processes
        mpi_tools.mpi_avg_grads(self.ac.pi.net)
        self.loss_pi_cost_before = loss_cost.item()
        b_flat = get_flat_gradients_from(self.ac.pi.net)

        p = g_flat - self.lagrangian_multiplier * b_flat
        x = conjugate_gradients(self.Fvp, p, self.cg_iters)
        assert torch.isfinite(x).all()
        pHp = torch.dot(x, self.Fvp(x))  # equivalent to : p^T x
        assert pHp.item() >= 0, 'No negative values.'

        # perform descent direction
        eps = 1.0e-8
        alpha = torch.sqrt(2 * self.target_kl / (pHp + eps))
        step_direction = alpha * x
        assert torch.isfinite(step_direction).all()
        ep_costs = self.logger.get_stats('EpCosts')[0]
        c = ep_costs - self.cost_limit
        c /= (self.logger.get_stats('EpLen')[0] + eps)  # rescale

        # determine step direction and apply SGD step after grads where set
        final_step_dir, accept_step = self.adjust_step_direction(
            step_dir=step_direction,
            g_flat=g_flat,
            p_dist=p_dist,
            data=data
        )
        # update actor network parameters
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self.ac.pi.net, new_theta)

        with torch.no_grad():
            q_dist = self.ac.pi.dist(data['obs'])
            kl = torch.distributions.kl.kl_divergence(p_dist,
                                                      q_dist).mean().item()
            loss_pi, pi_info = self.compute_loss_pi(data=data)

        self.logger.store(**{
            'Values/Adv': data['act'].numpy(),
            'Entropy': pi_info['ent'],
            'KL': kl,
            'PolicyRatio': pi_info['ratio'],
            'Loss/Pi': self.loss_pi_before,
            'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
            'Misc/AcceptanceStep': accept_step,
            'Misc/Alpha': alpha.item(),
            'Misc/StopIter': 1,
            'Misc/FinalStepNorm': final_step_dir.norm().item(),
            'Misc/xHx': pHp.item(),
            'Misc/gradient_norm': torch.norm(g_flat).item(),
            'Misc/cost_gradient_norm': torch.norm(b_flat).item(),
            'Misc/H_inv_g': x.norm().item(),
        })