import torch


def create_optimizer(model, c):
    if c.optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=c.learning_rate,
            weight_decay=c.weight_decay,
            betas=(c.adam_beta1, c.adam_beta2),
            eps=c.adam_epsilon,
        )
        return optimizer
    
    elif c.optimizer_name == "muon":
        try:
            from muon import Muon
        except ImportError:
            raise ImportError(
                "Muon optimizer not installed. Install with: pip install git+https://github.com/KellerJordan/Muon"
            )
        
        hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2]
        
        param_groups = [
            {
                'params': hidden_weights,
                'lr': c.muon_lr,
                'momentum': c.muon_momentum,
            },
            {
                'params': other_params,
                'lr': c.muon_auxiliary_lr,
            }
        ]
        
        optimizer = Muon(
            param_groups,
            lr=c.muon_lr,
            momentum=c.muon_momentum,
            nesterov=c.muon_nesterov,
            backend=c.muon_backend,
        )
        return optimizer
    
    elif c.optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=c.learning_rate,
            momentum=c.sgd_momentum,
            weight_decay=c.weight_decay,
            nesterov=c.sgd_nesterov,
        )
        return optimizer
    
    elif c.optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=c.learning_rate,
            betas=(c.adam_beta1, c.adam_beta2),
            eps=c.adam_epsilon,
            weight_decay=c.weight_decay,
        )
        return optimizer
    
    else:
        raise ValueError(f"Unknown optimizer: {c.optimizer_name}")

