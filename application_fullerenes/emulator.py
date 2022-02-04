#!/usr/bin/env python
import pickle
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bayesian_torch.layers import LinearReparameterization
from bayesian_torch.layers import LinearFlipout


prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -2.0

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -2.0

class BayesianNetwork(nn.Module):
    def __init__(self, x_dim=3, y_dim=4, hidden_dim=64):
        super(BayesianNetwork, self).__init__()

        self.blinear1 = LinearReparameterization(
                in_features=x_dim,
                out_features=hidden_dim,
                prior_mean=prior_mu,
                prior_variance=prior_sigma,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init,
        )
        self.blinear2 = LinearReparameterization(
                in_features=hidden_dim,
                out_features=hidden_dim,
                prior_mean=prior_mu,
                prior_variance=prior_sigma,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init,
        )
        self.blinear3 = LinearReparameterization(
                in_features=hidden_dim,
                out_features=hidden_dim,
                prior_mean=prior_mu,
                prior_variance=prior_sigma,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init,
        )
        self.blinear4 = LinearReparameterization(
                in_features=hidden_dim,
                out_features=y_dim,
                prior_mean=prior_mu,
                prior_variance=prior_sigma,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init,
        )

    def predict(self, x):
        ''' convenience for using the the same api as tf
        '''
        return self.forward(x)

    def forward(self, x):
        kl_sum = 0

        x_, kl = self.blinear1(x)
        kl_sum += kl
        x_ = F.relu(x_)

        #x_ = nn.Dropout(p=0.1)(x_)

        x_, kl = self.blinear2(x_)
        kl_sum += kl
        x_ = F.relu(x_)

        #x_ = nn.Dropout(p=0.1)(x_)

        x_, kl = self.blinear3(x_)
        kl_sum += kl
        x_ = F.relu(x_)

        x_, kl = self.blinear4(x_)
        kl_sum += kl

        return F.softmax(x_, dim=1), kl_sum

# helper functions
def run_experiment(param):
    c60 = param['c60_flow']
    sul = param['sultine_flow']
    T = param['T']

    x = np.array([[c60, sul, T]])

    _x  = feature_scaler.transform(x)
    pred, _ = model.predict(torch.tensor(_x).float())
    na, ma, ba, ta = pred.cpu().detach().numpy()[0]

    return na, ma, ba, ta

def eval_merit(param):
    na, ma, ba, ta = run_experiment(param)
    param['obj0'] = ba / ma  # obj0 =  BA / MA > 1
    param['obj1'] = ta   # obj1 = min TA

    # append also individual fractions
    param['NA'] = na
    param['MA'] = ma
    param['BA'] = ba
    param['TA'] = ta
    return param

if __name__ == '__main__':

    # cd
    pwd = os.getcwd()
    rootdir = "/".join(pwd.split('/')[:-1])
    optdir = pwd.split('/')[-1]
    os.chdir(rootdir)

    device = 'cpu'
    model = BayesianNetwork(3, 4, 64).to(device)
    checkpoint = 'torch_prod_models/fullerenes.pth'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    # load feature scaler
    feature_scaler = pickle.load(open('torch_prod_models/feature_scaler.pkl', 'rb'))

    # load in the param from gryffin
    param = pickle.load(open(f'{optdir}/param.pkl', 'rb'))
    os.remove(f'{optdir}/param.pkl')

    # make the observation
    observation = eval_merit(param)

    print('observation : ', observation)

    pickle.dump(observation, open(f'{optdir}/observation.pkl', 'wb'))
