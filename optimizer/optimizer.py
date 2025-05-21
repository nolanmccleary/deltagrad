import torch
from spectra.deltagrad.gradient_engine import NES_Engine #TODO: Change this back
from spectra.deltagrad.utils import anal_clamp



class Optimizer:

    def __init__(self, func_package: tuple, device_package: tuple, tensor: torch.tensor, **kwargs):
        self.func, self.loss_func, self.quant_func = func_package
        self.func_device, self.loss_func_device, self.quant_func_device = device_package
        self.tensor = tensor.clone()


    def get_delta(self, **kwargs):
        raise NotImplementedError("Subclasses must implement compute_gradient.")
    



class NES_Signed_Optimizer(Optimizer):

    def __init__(self, func_package, device_package, tensor, vecMin = None, vecMax = None):
        self.vecMin = vecMin
        self.vecMax = vecMax
        super().__init__(func_package, device_package, tensor)
        self.engine = NES_Engine(
            func=self.func, 
            loss_func=self.loss_func, 
            quant_func=self.quant_func, 
            func_device=self.func_device, 
            loss_func_device=self.loss_func_device, 
            quant_func_device=self.quant_func_device, 
            tensor=self.tensor)
    
    
    #Step coefficient encodes step size and direction 
    def get_delta(self, step_coeff, num_steps, perturbation_scale_factor, num_perturbations, beta=1, acceptance_func=None):

        tensor = self.tensor
        device = tensor.device

        vecMin = self.vecMin
        vecMax = self.vecMax    #Optimize for locality

        alpha = 1 - beta

        delta = torch.zeros_like(tensor)
        prev_step = torch.zeros_like(tensor)

        output_delta = delta    #Tensors shallow copy by default

        for _ in range(num_steps): 

            step = torch.sign(self.engine.compute_gradient(perturbation_scale_factor=perturbation_scale_factor, num_perturbations=num_perturbations, vecMin=self.vecMin, vecMax=self.vecMax)) 

            if vecMin is not None and vecMax is not None:
                
                safe_scale = anal_clamp(tensor, step, vecMin, vecMax, perturbation_scale_factor)
                
                step = step * safe_scale

            step = step * step_coeff * beta + prev_step * alpha
            prev_step = step

            tensor += step
            delta += step

            if acceptance_func is not None:
                if acceptance_func(tensor, delta):
                    output_delta = delta.clone()


        return output_delta
    



class NES_Optimizer(Optimizer):

    def __init__(self, func_package, device_package, tensor, vecMin = None, vecMax = None):
        self.vecMin = vecMin
        self.vecMax = vecMax
        super().__init__(func_package, device_package, tensor)
        self.engine = NES_Engine(
            func=self.func, 
            loss_func=self.loss_func, 
            quant_func=self.quant_func, 
            func_device=self.func_device, 
            loss_func_device=self.loss_func_device, 
            quant_func_device=self.quant_func_device, 
            tensor=self.tensor)
    
    
    #Step coefficient encodes step size and direction 
    def get_delta(self, step_coeff, num_steps, perturbation_scale_factor, num_perturbations, beta=1, acceptance_func=None):


        tensor = self.tensor
        device = tensor.device

        vecMin = self.vecMin
        vecMax = self.vecMax    #Optimize for locality

        alpha = 1 - beta

        delta = torch.zeros_like(tensor)
        prev_step = torch.zeros_like(tensor)

        output_delta = delta    #Tensors shallow copy by default

        for _ in range(num_steps): 

            step = self.engine.compute_gradient(perturbation_scale_factor=perturbation_scale_factor, num_perturbations=num_perturbations, vecMin=self.vecMin, vecMax=self.vecMax) #The step is very large so we have to normalize it
            tensorMax, idx = torch.abs(tensor).view(-1).max(0)
            stepMax, idy = torch.abs(step).view(-1).max(0)

            step_scaledown = (tensorMax / stepMax) 

            step *= step_scaledown


#            step_norm = step.norm()
#            tensor_norm = tensor.norm()
#            step *= (tensor_norm / (step_norm + EPS))


            if vecMin is not None and vecMax is not None:

                safe_scale = anal_clamp(tensor, step, vecMin, vecMax, perturbation_scale_factor)
                
                step = step * safe_scale

            step = (step * beta + prev_step * alpha) * step_coeff
            prev_step = step

            tensor += step
            delta += step

            if acceptance_func is not None:
                if acceptance_func(tensor, delta):
                    output_delta = delta.clone()


        return output_delta