"""
Lion Optimizer Plugin for Bhaskera.
Implements EvoLved Sign Momentum (Lion).
"""
import torch
from torch.optim.optimizer import Optimizer
from bhaskera.trainer.optimizer_registry import register_optimizer

class Lion(Optimizer):
    """
    Standard PyTorch implementation of the Lion optimizer.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Perform step weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                
                # Decay the momentum running average tracker
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

@register_optimizer("lion")
def build_lion(model, train_cfg):
    """
    Builder function called by the Bhaskera framework.
    We reuse Bhaskera's default parameter grouping to ensure 1-D tensors 
    (biases, LayerNorms) are properly excluded from weight decay.
    """
    from bhaskera.trainer.optim import _get_default_param_groups
    
    opt_cfg = train_cfg.optimizer
    
    # Extract kwargs, falling back to base training config if not specified
    lr = opt_cfg.kwargs.get("lr", train_cfg.lr)
    weight_decay = opt_cfg.kwargs.get("weight_decay", train_cfg.weight_decay)
    betas = opt_cfg.kwargs.get("betas", (0.9, 0.99))
    
    # Apply standard 2-D+ weight decay heuristic
    param_groups = _get_default_param_groups(model, weight_decay)
    
    return Lion(param_groups, lr=lr, betas=betas)
