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
        raise NotImplementedError(
            "Muon optimizer not yet implemented. "
            "Will be added in a future branch."
        )
    
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

