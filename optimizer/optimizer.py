import torch
from deltagrad.gradient_engine import NES_Engine


EPS = 1e-6


class Optimizer:

    def __init__(self, func_package: tuple, device_package: tuple, tensor: torch.tensor, **kwargs):
        self.func, self.delta_func, self.quant_func, self.loss_optimization_func = func_package
        self.func_device, self.delta_func_device, self.quant_func_device, self.loss_optimization_func_device = device_package
        self.tensor = tensor.clone()



    def get_delta(self, **kwargs):
        raise NotImplementedError("Subclasses must implement compute_gradient.")
    



class NES_Signed_Optimizer(Optimizer):

    def __init__(self, engine, func_package, device_package, tensor, vecMin = None, vecMax = None):
        self.vecMin = vecMin
        self.vecMax = vecMax
        super().__init__(engine, func_package, device_package, tensor)
        self.engine = NES_Engine(
            func=self.func, 
            delta_func=self.delta_func, 
            quant_func=self.quant_func, 
            func_device=self.func_device, 
            delta_func_device=self.delta_func_device, 
            quant_func_device=self.quant_func_device, 
            tensor=self.tensor)
    
    
    #Step coefficient encodes step size and direction 
    def get_delta(self, step_coeff, num_steps, perturbation_scale_factor, num_perturbations, beta=1, delta_threshold = None):
        original_func_output = None

        tensor = self.tensor
        device = tensor.device

        func = self.func
        delta_func = self.delta_func
        loss_optimization_func = self.loss_optimization_func

        func_device = self.func_device
        delta_func_device = self.delta_func_device
        loss_optimization_func_device = self.loss_optimization_func_device

        vecMin = self.vecMin
        vecMax = self.vecMax    #Optimize for locality

        alpha = 1 - beta

        delta = torch.zeros_like(tensor)
        output_delta = delta.clone()


        if delta_threshold is not None: 
            original_func_output = func(tensor.to(func_device))

        if loss_optimization_func is not None:
            min_loss = loss_optimization_func(tensor.to(loss_optimization_func_device))


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


            #REOWRK TODO: Generic representations for primary and secondary loss functions; This all needs to go
            if delta_threshold is not None:
                curr_func_output = func(tensor)
                func_delta = delta_func(original_func_output.to(delta_func_device), curr_func_output.to(delta_func_device))
                
                if func_delta >= delta_threshold:
                    curr_loss = loss_optimization_func(tensor.to(loss_optimization_func_device))  #Normalize secondary comparison as minimizing a loss function
                    if curr_loss < min_loss:
                        min_loss = curr_loss
                        output_delta = delta

            else:
                curr_loss = loss_optimization_func(tensor.to(loss_optimization_func_device))  #Normalize secondary comparison as minimizing a loss function
                if curr_loss < min_loss:
                    min_loss = curr_loss
                    output_delta = delta


 
        return output_delta