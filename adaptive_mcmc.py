import torch
import numpy as np
from tqdm import tqdm
from torch.distributions.gamma import Gamma
from torch.linalg import solve
from torch import einsum, eye, randn_like, ones_like, stack,svd

def initialization(X, n, r):
    
    U, S, _ = svd(X.T / (n-1)**0.5)

    sigma2_estimator = S[r:].square().sum() / (n-r)

    mu = U[:, 0:r] * (S[0:r].square() - sigma2_estimator).sqrt()

    return mu, sigma2_estimator

def sample_eta(X, B, sigma ,device):
    
    _, r = B.shape

    C = B.T / sigma

    mu = C @ (X / sigma).T

    eta = torch.linalg.solve(C @ C.T + torch.eye(r, device=device, dtype=torch.float64), mu + C @ torch.randn_like(X).T + torch.randn_like(mu))
    
    return eta.T

def sample_delta(B, delta, tau, a1, a2):

    p,r = B.shape
    
    ink = (B.square() * tau).sum(0)

    for j in range(0, r):
        
        lam = torch.cumprod(delta, dim = 0)
        
        if j == 0:
            delta[j] = Gamma(a1 + 0.5 * p * (r - j),  0.5 * ((lam[j:] / delta[j]) * ink [j:]).sum() + 1).sample()
        else:
            delta[j] = Gamma(a2 + 0.5 * p * (r - j),  0.5 * ((lam[j:] / delta[j]) * ink [j:]).sum() + 1).sample()
                
    return delta

def Gibbs_sampling(X, r=50, M=10000, burn_in = 15000, start_adapt = 500):

    ## set hyperparameters
    
    a_sigma = 1
    b_sigma = 1
    v = 3
    a1 = 3
    a2 = 3
    alpha0 = -1
    alpha1 = -5 * 10**(-4)
    eposilon = 10 ** (-4)

    N, P = X.shape
    r_estimate = 0
    rmax = P+1
    r = int(5 * np.log(P))
    
    X = X.to(torch.float64)
    device = X.device

    cov_estimate = torch.zeros(P, P, device=device, dtype=torch.float64)


    ## initialization

    B_sample, sigma2_sample = initialization(X, N, r)
    sigma2_sample = sigma2_sample.repeat(P)
    delta_sample = torch.ones(r, device = device, dtype = torch.float64)
    lam_sample = torch.cumprod(delta_sample, dim=0)

    for i in tqdm(range(1, M + burn_in)):
        
        # sample eta
        eta_sample = sample_eta(X, B_sample, sigma2_sample.sqrt(), device)

        # sample sigma2
        sigma2_sample = (b_sigma + 0.5 * ((X - eta_sample @ B_sample.T).pow(2).sum(0))) / Gamma(a_sigma + 0.5 * N, ones_like(sigma2_sample)).sample()

        # sample shrinkage parameter
        # sample tau
        tau_sample = Gamma(0.5 * (v + 1), 0.5 * (v + B_sample.square() * lam_sample)).sample()

        #sample lam
        delta_sample = sample_delta(B_sample, delta_sample, tau_sample, a1, a2)
        lam_sample = torch.cumprod(delta_sample, dim=0)

        D = 1 / (tau_sample * lam_sample).sqrt()

        # sample B
        C = (D.view(P,r,1) * eta_sample.T.view(1,r,N))/ sigma2_sample.sqrt().view(P,1,1)

        b =  D * (eta_sample.T @ X / sigma2_sample).T + torch.einsum('bij,bj->bi', C, randn_like(X.T)) + randn_like(B_sample)

        B_sample = D * solve(einsum('bij,bjk->bik', C, C.transpose(1,2)) + eye(r, device= device, dtype= torch.float64).view(1, r, r), b)

        if (i + 1) > burn_in:

            cov_sample = B_sample @ B_sample.T + torch.diag(sigma2_sample)
            
            cov_estimate = cov_sample / (i+1-burn_in) + (i-burn_in) / (i+1-burn_in) * cov_estimate

            r_estimate = r / (i+1-burn_in) + (i-burn_in) / (i+1-burn_in) * r_estimate

        
        if (i + 1) > start_adapt:
            
            u = torch.rand([1], device=device, dtype=torch.float64)
            
            if u <= torch.exp(torch.tensor([alpha0 + alpha1 * (i + 1)], device=device, dtype=torch.float64)):
                
                # active factor number
                max_col, _ = B_sample.abs().max(dim=0)

                active = max_col >= eposilon

                r_star = int(active.double().sum().item())

                # reduce truncation
                if r_star < r:

                    r = r_star

                    if r_star == 0:
                        print("error!")

                    B_sample = B_sample[:, active]

                    delta_sample = delta_sample[active]                    
                    
                    lam_sample = torch.cumprod(delta_sample, dim=0)
                    
                 
                # extend truncation
                elif r_star < rmax:

                    r = r + 1

                    delta_sample = torch.cat([delta_sample, Gamma(torch.tensor([a2], device=device, dtype=torch.float64), 1).sample()])

                    lam_sample = torch.cumprod(delta_sample, dim=0)

                    tau_add = Gamma(v / 2, torch.tensor([v / 2], device=device, dtype=torch.float64).repeat(P)).sample()

                    D_add = torch.diag(1 / (tau_add * lam_sample[-1]).sqrt())

                    B_sample = torch.cat([B_sample,  (D_add @ torch.randn(P, device=device, dtype=torch.float64)).unsqueeze(1)], dim=1)



        
    return cov_estimate.squeeze().to('cpu'), r_estimate