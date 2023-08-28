import torch

from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods
from mpc.env_dx import cartpole

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

import os
import io
import base64
import tempfile
from IPython.display import HTML

from tqdm import tqdm

dx = cartpole.CartpoleDx()

n_batch, T, mpc_T = 2, 50, 25


def uniform(shape, low, high):
    r = high - low
    return torch.rand(shape) * r + low


torch.manual_seed(0)
th = uniform(n_batch, -2 * np.pi, 2 * np.pi)
thdot = uniform(n_batch, -.5, .5)
x = uniform(n_batch, -0.5, 0.5)
xdot = uniform(n_batch, -0.5, 0.5)
xinit = torch.stack((x, xdot, torch.cos(th), torch.sin(th), thdot), dim=1)

x = xinit
u_init = None

q, p = dx.get_true_obj()
Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1, 1)
p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)

action_history = []
for t in tqdm(range(T)):
    ctrl = mpc.MPC(
        dx.n_state, dx.n_ctrl, mpc_T,
        u_init=u_init,
        u_lower=dx.lower, u_upper=dx.upper,
        lqr_iter=50,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        linesearch_decay=dx.linesearch_decay,
        max_linesearch_iter=dx.max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF,
        eps=1e-2,)
    nominal_states, nominal_actions, nominal_objs = ctrl(x, QuadCost(Q, p), dx)

    next_action = nominal_actions[0]
    action_history.append(next_action)
    u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, dx.n_ctrl)), dim=0)
    u_init[-2] = u_init[-3]
    x = dx(x, next_action)


action_history = torch.stack(action_history).detach()[:, :, 0]