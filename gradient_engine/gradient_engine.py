import torch
from spectra.deltagrad.utils import generate_perturbation_vectors, anal_clamp


EPS = 1e-6 #1 in a million, Baby
class Gradient_Engine:

    def __init__(self, func, loss_func, quant_func, func_device, loss_func_device, quant_func_device, tensor):    
        self.func = func
        self.loss_func = loss_func
        self.quant_func = quant_func
        self.func_device = func_device
        self.loss_func_device = loss_func_device
        self.quant_func_device = quant_func_device
        self.tensor = tensor
        self.gradient = torch.zeros_like(self.tensor)
        

    def compute_gradient(self, **kwargs):
        raise NotImplementedError("Subclasses must implement compute_gradient.")




class NES_Engine(Gradient_Engine):
    
    def __init__(self, func, loss_func, quant_func, func_device, loss_func_device, quant_func_device, tensor):      
        super().__init__(func, loss_func, quant_func, func_device, loss_func_device, quant_func_device, tensor)


    def compute_gradient(self, perturbation_scale_factor, num_perturbations, vecMin=None, vecMax=None):
        tensor = self.tensor    #Optimize for locality
        func = self.func
        loss_func = self.loss_func
        quant_func = self.quant_func
        func_device = self.func_device
        loss_func_device = self.loss_func_device
        quant_func_device = self.quant_func_device

        device = tensor.device

        perturbations = generate_perturbation_vectors(num_perturbations, tensor.shape, device) #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]
        last_output = func(tensor).to(loss_func_device)

        if vecMin is not None and vecMax is not None:
            
            safe_scale = anal_clamp(tensor, perturbations, vecMin, vecMax, perturbation_scale_factor)

            cand_batch = (tensor + perturbations * safe_scale).to(func_device).clamp(vecMin, vecMax) #[t1, t2, t3] + [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]] -> [[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]] where cxy = t[y] + p[x,y]
        
        else:
            cand_batch = (tensor + perturbations).to(func_device)    #Clampage sorta redundant here but better safe than sorry

        if quant_func is not None:
            cand_batch = quant_func(cand_batch.to(quant_func_device))

        new_outputs = func(cand_batch).to(loss_func_device) 
        
        diffs = loss_func(new_outputs, last_output)       
        deltas = diffs.sum(dim=1).to(tensor.dtype).view(num_perturbations, *((1,) * tensor.dim()))

        return (deltas * perturbations.to(device)).sum(dim=0).to(device).view(tensor.shape)  #[d1, d2, d3] -> VecSum([[d1], [d2], [d3]] * [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]) -> [g1, g2, g3] where gx = [dx] * [px1, px2, px3]