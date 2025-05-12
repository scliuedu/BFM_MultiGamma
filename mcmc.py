import torch
from tqdm import tqdm
from torch.distributions.gamma import Gamma


# define product function with one element out

def prod_oneout(delta, t, k):
    
    """
    t: order of the out element
    k: product of the first k elements
    """

    prod = 1
    
    for i in range(k):
        if i != t-1:
            prod = prod * delta[i]
    
    return prod.item()

def sample_eta(X, B, sigma ,device):
    
    _, r = B.shape

    C = B.T / sigma

    mu = C @ (X / sigma).T

    eta = torch.linalg.solve(C @ C.T + torch.eye(r, device=device, dtype=torch.float64), mu + C @ torch.randn_like(X).T + torch.randn_like(mu))
    
    return eta.T

def sample_tau(B, lam, v):
    
    tau = Gamma(0.5 * (v + 1), 0.5 * (v + B.pow(2) * lam)).sample()

    return tau

def sample_delta(B, delta, tau, a1, a2):

    P,r = B.size()
    delta_samplein = delta.clone()

    # update delta1
    prod_out1 = torch.ones(r-1, device = B.device, dtype = torch.float64)
    
    for k in range(len(prod_out1)):
        prod_out1[k] = prod_oneout(delta_samplein, 1, k+2)

    B_cap = B[:, 1:].clone()
    tau_cap = tau[:, 1:].clone()
    
    delta1 = Gamma(a1 + 0.5 * P * r, 1 + 0.5 * (B_cap.pow(2) * tau_cap * prod_out1).sum()).sample()
    delta_samplein[0] = delta1

    #update other delta
    for t in range(2, r+1):
        prod_out2 = torch.ones(r-t, device = B.device, dtype = torch.float64)

        for k in range(len(prod_out2)):
            prod_out2[k] = prod_oneout(delta_samplein, t, k+t+1)

        B_cap2 = B[:, t:].clone()
        tau_cap2 = tau[:, t:].clone()

        delta2 = Gamma(a1 + 0.5 * P * (r-t+1), 1 + 0.5 * (B_cap2.pow(2) * tau_cap2 * prod_out2).sum()).sample()
        delta_samplein[t-1] = delta2

    return delta_samplein

def Gibbs_sampling(X, r=50, M=1000, burn_in = 1500):

    ## set hyperparameters
    
    a_sigma = 0.5
    b_sigma = 0.5
    v = 2
    a1 = 2
    a2 = 2

    N, P = X.shape
    
    X = X.to(torch.float64)
    device = X.device

    B_samples = []
    sigma2_samples = []

    ## initialization

    B_sample = torch.ones(P, r, device = device, dtype = torch.float64)
    sigma2_sample = torch.ones(P, device = device, dtype = torch.float64)
    delta_sample = torch.ones(r, device = device, dtype = torch.float64)
    lam_sample = torch.cumprod(delta_sample, dim=0)

    for i in tqdm(range(1, M + burn_in)):
        
        # sample eta
        eta_sample = sample_eta(X, B_sample, sigma2_sample.sqrt(), device)

        # sample sigma2
        sigma2_sample = (b_sigma + 0.5 * ((X - eta_sample @ B_sample.T).pow(2).sum(0))) / Gamma(a_sigma + 0.5 * N, torch.ones(P, device = device, dtype = torch.float64)).sample()

        # sample shrinkage parameter
        # sample tau
        tau_sample = sample_tau(B_sample, lam_sample, v)

        #sample lam
        delta_sample = sample_delta(B_sample, delta_sample, tau_sample, a1, a2)
        lam_sample = torch.cumprod(delta_sample, dim=0)

        D = 1 / (tau_sample * lam_sample).sqrt()

        # sample B
        C = (D.unsqueeze(-1) * eta_sample.T.unsqueeze(0)) / sigma2_sample.sqrt().unsqueeze(-1).unsqueeze(-1)

        b =  D * (eta_sample.T @ X / sigma2_sample).T + torch.einsum('bij,bj->bi', C, torch.randn(P, N, device= device, dtype= torch.float64)) + torch.randn(P, r, device= device, dtype = torch.float64)

        B_sample = D * torch.linalg.solve(torch.einsum('bij,bjk->bik', C, C.transpose(1,2)) + torch.eye(r, device= device, dtype= torch.float64).unsqueeze(0).repeat(P, 1, 1), b)

        if (i + 1) > burn_in:

            B_samples.append(B_sample)
            sigma2_samples.append(sigma2_sample)

        
    return torch.stack(B_samples).squeeze().to('cpu'), torch.stack(sigma2_samples).squeeze().to('cpu')