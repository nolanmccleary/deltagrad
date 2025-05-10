import torch
from spectra.deltagrad.gradient_engine import NES_Engine #TODO: Change this back


EPS = 1e-6


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

        for _ in range(num_steps): #The loop seems greasy because it is. However, when there is a lot of meat there is bound to be some grease. It's a good thing that most of this meat lies within the repeated function calls during gradient estimation. 

            step = torch.sign(self.engine.compute_gradient(perturbation_scale_factor=perturbation_scale_factor, num_perturbations=num_perturbations, vecMin=self.vecMin, vecMax=self.vecMax)) 

            if vecMin is not None and vecMax is not None:
                pos_scale = torch.where(
                    step > 0,
                    (vecMax - tensor) / (step + EPS),
                    torch.tensor(perturbation_scale_factor, device=device),
                )
                neg_scale = torch.where(                                       
                    tensor < 0,
                    (vecMin - tensor) / (step - EPS),
                    torch.tensor(perturbation_scale_factor, device=device),
                )
                safe_scale = torch.min(pos_scale, neg_scale).clamp(max=1.0)
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
    def get_delta(self, step_coeff, num_steps, perturbation_scale_factor, num_perturbations, gradient_scale_factor=1, beta=1, acceptance_func=None):


        tensor = self.tensor
        device = tensor.device

        vecMin = self.vecMin
        vecMax = self.vecMax    #Optimize for locality

        alpha = 1 - beta

        delta = torch.zeros_like(tensor)
        prev_step = torch.zeros_like(tensor)

        output_delta = delta    #Tensors shallow copy by default

        for _ in range(num_steps): #The loop seems greasy because it is. However, when there is a lot of meat there is bound to be some grease. It's a good thing that most of this meat lies within the repeated function calls during gradient estimation. 

            step = self.engine.compute_gradient(perturbation_scale_factor=perturbation_scale_factor, num_perturbations=num_perturbations, vecMin=self.vecMin, vecMax=self.vecMax) #The step is very large so we have to normalize it
            tensorMax, idx = torch.abs(tensor).view(-1).max(0)
            stepMax, idy = torch.abs(step).view(-1).max(0)

            step_scaledown = (tensorMax / stepMax) * gradient_scale_factor

            step *= step_scaledown

            if vecMin is not None and vecMax is not None:
                pos_scale = torch.where(
                    step > 0,
                    (vecMax - tensor) / (step + EPS),
                    torch.tensor(perturbation_scale_factor, device=device),
                )
                neg_scale = torch.where(   
                    tensor < 0,
                    (vecMin - tensor) / (step - EPS),
                    torch.tensor(perturbation_scale_factor, device=device),
                )
                safe_scale = torch.min(pos_scale, neg_scale).clamp(max=1.0)
                step = step * safe_scale

            step = step * step_coeff * beta + prev_step * alpha
            prev_step = step

            tensor += step
            delta += step

            if acceptance_func is not None:
                if acceptance_func(tensor, delta):
                    output_delta = delta.clone()


        return output_delta