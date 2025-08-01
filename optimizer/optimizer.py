import torch
from spectra.deltagrad.gradient_engine import NES_Engine, Gradient_Engine_Config
from spectra.deltagrad.utils import anal_clamp
from pydantic import BaseModel, Field
from typing import Callable, Optional

class Optimizer_Config(BaseModel):
    func: Callable = Field(..., description="Function")
    loss_func: Callable = Field(..., description="Loss function")
    quant_func: Callable = Field(..., description="Quantization function")
    func_device: str = Field(..., description="Function device")
    loss_func_device: str = Field(..., description="Loss function device")
    quant_func_device: str = Field(..., description="Quantization function device")
    verbose: str = Field(..., description="Verbosity level")

class Delta_Config(BaseModel):
    step_coeff: float = Field(..., description="Step coefficient")
    num_steps: int = Field(..., description="Number of steps")
    perturbation_scale_factor: float = Field(..., description="Perturbation scale factor")
    num_perturbations: int = Field(..., description="Number of perturbations")
    beta: float = Field(..., description="Beta")
    acceptance_func: Callable = Field(..., description="Acceptance function")
    vecMin: Optional[float] = Field(default=None, description="Vector minimum")
    vecMax: Optional[float] = Field(default=None, description="Vector maximum")


class Optimizer:

    def __init__(self, config: Optimizer_Config):
        self.config = config

    def log(self, msg: str) -> None:
        """Log message if verbose mode is enabled"""
        if self.config.verbose == "on":
            print(msg)

    def get_delta(self, **kwargs):
        raise NotImplementedError("Subclasses must implement get_delta.")
    


class NES_Signed_Optimizer(Optimizer):

    def __init__(self, config: Optimizer_Config):
        super().__init__(config)
        
        self.gradient_engine_config = Gradient_Engine_Config(
            func=self.config.func,
            loss_func=self.config.loss_func,
            quant_func=self.config.quant_func,
            func_device=self.config.func_device,
            loss_func_device=self.config.loss_func_device,
            quant_func_device=self.config.quant_func_device,
            verbose=self.config.verbose)
        
        self.engine = NES_Engine(self.gradient_engine_config)
        
        
    def get_delta(self, tensor: torch.Tensor, config: Delta_Config) -> tuple[int, torch.Tensor, bool]:
        working_tensor  = tensor.clone()

        beta = config.beta
        vecMin = config.vecMin
        vecMax = config.vecMax
        perturbation_scale_factor = config.perturbation_scale_factor
        num_perturbations = config.num_perturbations
        step_coeff = config.step_coeff
        acceptance_func = config.acceptance_func

        alpha           = 1 - beta

        delta           = torch.zeros_like(working_tensor)
        prev_step       = torch.zeros_like(working_tensor)


        accepted = False

        step_count = 0

        for _ in range(config.num_steps): 
            step_count      += 1

            self.log(f"\nStep {step_count}")

            step            = torch.sign(self.engine.compute_gradient(tensor=working_tensor, perturbation_scale_factor=perturbation_scale_factor, num_perturbations=num_perturbations))
            step            = (step * beta + prev_step * alpha)
            base_step       = step.clone()

            self.log(f"Pre-clamp step: abs max={torch.max(torch.abs(step)):.6f}, abs min={torch.min(torch.abs(step)):.6f}, mean={torch.mean(step):.6f}, rms mean={torch.sqrt(torch.mean(step**2)):.6f}, abs mean={torch.mean(torch.abs(step)):.6f}")
            if vecMin is not None and vecMax is not None:
                safe_scale  = anal_clamp(working_tensor, step, vecMin, vecMax, step_coeff)
                step        = step * safe_scale
            
            self.log(f"Post-clamp step: abs max={torch.max(torch.abs(step)):.6f}, abs min={torch.min(torch.abs(step)):.6f}, mean={torch.mean(step):.6f}, rms mean={torch.sqrt(torch.mean(step**2)):.6f}, abs mean={torch.mean(torch.abs(step)):.6f}")
            self.log(f"Step cosine similarity after clamp: {torch.cosine_similarity(step.flatten(), base_step.flatten(), dim=0):.10f}\n")

            prev_step       = step

            working_tensor  += step
            delta           += step

            break_loop, accepted = acceptance_func(working_tensor, step_count)
            if break_loop:
                break

        return step_count, delta, accepted



class NES_Optimizer(Optimizer):

    def __init__(self, config: Optimizer_Config):
        super().__init__(config)
        
        self.gradient_engine_config = Gradient_Engine_Config(
            func=self.config.func,
            loss_func=self.config.loss_func,
            quant_func=self.config.quant_func,
            func_device=self.config.func_device,
            loss_func_device=self.config.loss_func_device,
            quant_func_device=self.config.quant_func_device,
            verbose=self.config.verbose)
        self.engine = NES_Engine(self.gradient_engine_config)
    
    
    
    def get_delta(self, tensor: torch.Tensor, config: Delta_Config) -> tuple[int, torch.Tensor, bool]:
        working_tensor  = tensor.clone()

        beta = config.beta
        vecMin = config.vecMin
        vecMax = config.vecMax
        perturbation_scale_factor = config.perturbation_scale_factor
        num_perturbations = config.num_perturbations
        step_coeff = config.step_coeff
        acceptance_func = config.acceptance_func

        alpha           = 1 - beta

        delta           = torch.zeros_like(working_tensor)
        prev_step       = torch.zeros_like(working_tensor)

        accepted = False

        step_count = 0

        for _ in range(config.num_steps): 
            step_count      += 1

            self.log(f"\nStep {step_count}")

            step            = self.engine.compute_gradient(tensor=working_tensor, perturbation_scale_factor=perturbation_scale_factor, num_perturbations=num_perturbations)
            step            = (step * beta + prev_step * alpha)
            base_step       = step.clone()

            self.log(f"Pre-clamp step: max={torch.max(step):.6f}, min={torch.min(step):.6f}, mean={torch.mean(step):.6f}, rms mean={torch.sqrt(torch.mean(step**2)):.6f}, abs mean={torch.mean(torch.abs(step)):.6f}")
            if vecMin is not None and vecMax is not None:
                safe_scale  = anal_clamp(working_tensor, step, vecMin, vecMax, step_coeff)
                step        = step * safe_scale
            
            self.log(f"Post-clamp step: max={torch.max(step):.6f}, min={torch.min(step):.6f}, mean={torch.mean(step):.6f}, rms mean={torch.sqrt(torch.mean(step**2)):.6f}, abs mean={torch.mean(torch.abs(step)):.6f}")
            self.log(f"Step cosine similarity after clamp: {torch.cosine_similarity(step.flatten(), base_step.flatten(), dim=0):.10f}")

            prev_step       = step

            working_tensor  += step
            delta           += step


            break_loop, accepted = acceptance_func(working_tensor, step_count)
            if break_loop:
                break

        return step_count, delta, accepted



class Colinear_Optimizer(Optimizer):

    def __init__(self, config: Optimizer_Config):
        super().__init__(config)
        
        self.gradient_engine_config = Gradient_Engine_Config(
            func=self.config.func,
            loss_func=self.config.loss_func,
            quant_func=self.config.quant_func,
            func_device=self.config.func_device,
            loss_func_device=self.config.loss_func_device,
            quant_func_device=self.config.quant_func_device,
            verbose=self.config.verbose)
        self.engine = NES_Engine(self.gradient_engine_config)
    
    
    def get_delta(self, tensor: torch.Tensor, config: Delta_Config) -> tuple[int, torch.Tensor, bool]:
        working_tensor  = tensor.clone()
        perturbation_scale_factor = config.perturbation_scale_factor
        num_perturbations = config.num_perturbations
        step_coeff = config.step_coeff
        acceptance_func = config.acceptance_func

        accepted = False

        step_count = 0
        init_step = self.engine.compute_gradient(tensor=working_tensor, perturbation_scale_factor=perturbation_scale_factor, num_perturbations=num_perturbations) * step_coeff
        delta = init_step.clone()
        
        for _ in range(config.num_steps): 

            delta += init_step

            step_count += 1

            self.log(f"\nStep {step_count}")
            self.log(f"Delta max: {torch.max(delta):.6f}, min: {torch.min(delta):.6f}, mean: {torch.mean(delta):.6f}, rms mean: {torch.sqrt(torch.mean(delta**2)):.6f}, abs mean: {torch.mean(torch.abs(delta)):.6f}")


            break_loop, accepted = acceptance_func(torch.add(working_tensor, delta), step_count)
            if break_loop:
                break

        return step_count, delta, accepted



class Gaussian_Optimizer(Optimizer):

    def __init__(self, config: Optimizer_Config):
        super().__init__(config)
        
        self.gradient_engine_config = Gradient_Engine_Config(
            func=self.config.func,
            loss_func=self.config.loss_func,
            quant_func=self.config.quant_func,
            func_device=self.config.func_device,
            loss_func_device=self.config.loss_func_device,
            quant_func_device=self.config.quant_func_device,
            verbose=self.config.verbose)
        self.engine = NES_Engine(self.gradient_engine_config)
    
    
    
    def get_delta(self, tensor: torch.Tensor, config: Delta_Config) -> tuple[int, torch.Tensor, bool]:
        working_tensor  = tensor.clone()

        beta = config.beta
        vecMin = config.vecMin
        vecMax = config.vecMax
        perturbation_scale_factor = config.perturbation_scale_factor
        num_perturbations = config.num_perturbations
        step_coeff = config.step_coeff
        acceptance_func = config.acceptance_func

        delta           = torch.zeros_like(working_tensor)

        accepted = False

        step_count = 0

        for _ in range(config.num_steps): 
            step_count      += 1

            step = torch.randn_like(working_tensor) * step_coeff

            working_tensor  += step
            delta           += step

            self.log(f"\nStep {step_count}")
            self.log(f"Delta max: {torch.max(delta):.6f}, min: {torch.min(delta):.6f}, mean: {torch.mean(delta):.6f}, rms mean: {torch.sqrt(torch.mean(delta**2)):.6f}, abs mean: {torch.mean(torch.abs(delta)):.6f}")


            break_loop, accepted = acceptance_func(working_tensor, step_count)
            if break_loop:
                break

        return step_count, delta, accepted