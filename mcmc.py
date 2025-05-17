import torch
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

def Gibbs_sampling(X, r=50, M=1000, burn_in = 1500):

    ## set hyperparameters
    
    a_sigma = 1
    b_sigma = 1
    v = 3
    a1 = 3
    a2 = 3

    N, P = X.shape
    
    X = X.to(torch.float64)
    device = X.device

    B_samples = []
    sigma2_samples = []

    ## initialization

    # B_sample = torch.ones(P, r, device = device, dtype = torch.float64)
    # sigma2_sample = torch.ones(P, device = device, dtype = torch.float64)
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

            B_samples.append(B_sample)
            sigma2_samples.append(sigma2_sample)

        
    return stack(B_samples).squeeze().to('cpu'), stack(sigma2_samples).squeeze().to('cpu')