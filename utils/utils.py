import torch

EPS = 1e-6


def generate_perturbation_vectors(num_perturbations, shape, device):
    base    = torch.randn((num_perturbations // 2, *shape), dtype=torch.float32, device=device) 
    absmax  = base.abs().amax(dim=1, keepdim=True)
    scale   = torch.where(absmax > 0, 1.0 / absmax, torch.tensor(1.0, device = device)) #scale tensor to get max val of each generated perturbation so that we can normalize
    base    = base * scale
    return torch.cat([base, -base], dim=0) #Mirror to preserve distribution



#Analytic Clamp
def anal_clamp(tensor, step, vecMin, vecMax, clamp_max=1.0):
    assert clamp_max <= 1.0, "ERROR: Clamp should not boost perturbation primitive!"
    
    device              = tensor.device
    
    pos_scale = torch.where(
        step > 0,
        (vecMax - tensor) / (step + EPS),
        torch.tensor(clamp_max, device=device),
    )
    neg_scale = torch.where(   
        step < 0,
        (vecMin - tensor) / (step - EPS),
        torch.tensor(clamp_max, device=device),
    )
    
    safe_scale_vector   = torch.min(pos_scale, neg_scale).clamp(min=0.0, max=clamp_max)
    return safe_scale_vector