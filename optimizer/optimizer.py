import torch
from spectra.deltagrad.gradient_engine import NES_Engine #TODO: Change this back
from spectra.deltagrad.utils import anal_clamp



class Optimizer:

    def __init__(self, func_package: tuple, device_package: tuple, tensor: torch.tensor, **kwargs):
        self.func, self.loss_func, self.quant_func                      = func_package
        self.func_device, self.loss_func_device, self.quant_func_device = device_package
        self.tensor                                                     = tensor.clone()


    def get_delta(self, **kwargs):
        raise NotImplementedError("Subclasses must implement compute_gradient.")
    


class NES_Signed_Optimizer(Optimizer):

    def __init__(self, func_package, device_package, tensor, vecMin = None, vecMax = None):
        self.vecMin = vecMin
        self.vecMax = vecMax
        super().__init__(func_package, device_package, tensor)
        self.engine = NES_Engine(
            func                = self.func, 
            loss_func           = self.loss_func, 
            quant_func          = self.quant_func, 
            func_device         = self.func_device, 
            loss_func_device    = self.loss_func_device, 
            quant_func_device   = self.quant_func_device, 
            tensor=self.tensor)
    
    
    
    def get_delta(self, step_coeff, num_steps, perturbation_scale_factor, num_perturbations, beta=1, acceptance_func=None) -> torch.Tensor:
        tensor          = self.tensor.clone()

        vecMin          = self.vecMin
        vecMax          = self.vecMax    

        alpha           = 1 - beta

        delta           = torch.zeros_like(tensor)
        prev_step       = torch.zeros_like(tensor)

        if acceptance_func is None:
            accepted = True
        
        else:
            accepted = False

        step_count = 0

        for _ in range(num_steps): 
            step_count += 1

            step            = torch.sign(self.engine.compute_gradient(perturbation_scale_factor=perturbation_scale_factor, num_perturbations=num_perturbations, vecMin=self.vecMin, vecMax=self.vecMax))
            step            = (step * beta + prev_step * alpha)

            if vecMin is not None and vecMax is not None:
                safe_scale  = anal_clamp(tensor, step, vecMin, vecMax, step_coeff)
                step        = step * safe_scale

            prev_step       = step

            tensor          += step
            delta           += step

            if acceptance_func is not None:
                break_loop, accepted = acceptance_func(tensor, step_count)
                if break_loop:
                    break

        return step_count, delta, accepted



class NES_Optimizer(Optimizer):

    def __init__(self, func_package, device_package, tensor, vecMin = None, vecMax = None):
        self.vecMin = vecMin
        self.vecMax = vecMax
        super().__init__(func_package, device_package, tensor)
        self.engine = NES_Engine(
            func                = self.func, 
            loss_func           = self.loss_func, 
            quant_func          = self.quant_func, 
            func_device         = self.func_device, 
            loss_func_device    = self.loss_func_device, 
            quant_func_device   = self.quant_func_device, 
            tensor              = self.tensor)
    
    
    
    def get_delta(self, step_coeff, num_steps, perturbation_scale_factor, num_perturbations, beta=1, acceptance_func=None) -> torch.Tensor:
        tensor          = self.tensor.clone()
        device          = tensor.device

        vecMin          = self.vecMin
        vecMax          = self.vecMax    

        alpha           = 1 - beta

        delta           = torch.zeros_like(tensor)
        prev_step       = torch.zeros_like(tensor)

        if acceptance_func is None:
            accepted = True
        
        else:
            accepted = False

        step_count = 0

        for _ in range(num_steps): 
            step_count      += 1

            step            = self.engine.compute_gradient(perturbation_scale_factor=perturbation_scale_factor, num_perturbations=num_perturbations, vecMin=self.vecMin, vecMax=self.vecMax) * step_coeff
            step            = (step * beta + prev_step * alpha)

            '''
            if vecMin is not None and vecMax is not None:
                safe_scale  = anal_clamp(tensor, step, vecMin, vecMax, step_coeff)
                step        = step * safe_scale
            '''

            prev_step       = step

            tensor          += step
            delta           += step


            if acceptance_func is not None:
                break_loop, accepted = acceptance_func(tensor, step_count)
                if break_loop:
                    break

        return step_count, delta, accepted