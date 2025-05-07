import torch
from .utils import generate_perturbation_vectors


EPS = 1e-6 #1 in a million baby
class Gradient_Engine:

    def __init__(self, func, delta_func, tensor, device, func_device, num_perturbations, quant_func=None):    
        self.func = func
        self.delta_func = delta_func
        self.quant_func = quant_func
        self.device = device
        self.func_device = func_device
        self.num_perturbations = num_perturbations
        self.tensor = tensor.to(self.device)
        self.gradient = torch.zeros_like(self.tensor)
        


    def compute_gradient(self, perturbation_scale_factor, vecMin=None, vecMax=None):
        perturbations = generate_perturbation_vectors(self.num_perturbations, self.tensor.shape, self.device) #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]
        last_output = self.func(self.tensor)

        if vecMin is not None and vecMax is not None:
            pos_scale = torch.where(
                perturbations > 0,
                (vecMax - self.tensor) / (perturbations + EPS),
                torch.tensor(perturbation_scale_factor, device=self.device),
            )
            neg_scale = torch.where(
                perturbations < 0,
                (vecMin - self.tensor) / (perturbations - EPS),
                torch.tensor(perturbation_scale_factor, device=self.device),
            )
            safe_scale = torch.min(torch.tensor(1.0), torch.min(pos_scale, neg_scale))
            cand_batch = (self.tensor + perturbations * safe_scale).to(self.func_device).clamp(vecMin, vecMax) #[t1, t2, t3] + [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]] -> [[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]] where cxy = t[y] + p[x,y]
        
        else:
            cand_batch = (self.tensor + perturbations).to(self.func_device)    #Clampage sorta redundant here but better safe than sorry

        if self.quant_func is not None:
            cand_batch = self.quant_func(cand_batch)

        new_outputs = torch.stack([self.func(v) for v in cand_batch], dim=0).to(self.device) #[NUM_PERTURBATIONS, N_BITS]
        
        diffs = self.delta_func(new_outputs, last_output)
        deltas = diffs.sum(dim=1).to(self.tensor.dtype).view(self.num_perturbations, *((1,) * self.tensor.dim()))

        gradient = (deltas * perturbations.to(self.device)).sum(dim=0).to(self.device).view(self.tensor.shape)  #[d1, d2, d3] -> VecSum([[d1], [d2], [d3]] * [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]) -> [g1, g2, g3] where gx = [dx] * [px1, px2, px3]
        return gradient