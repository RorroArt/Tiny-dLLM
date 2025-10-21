from sws import Config


def get_config():
    c = Config()
    
    c.model_name = "answerdotai/ModernBERT-base"
    
    c.dataset_name = "wikitext"
    c.dataset_config = "wikitext-2-raw-v1"
    
    c.n_steps = 10
    c.num_epochs = 30
    c.batch_size = 16
    c.max_len = 256
    c.prefix_len = 16
    
    c.optimizer_name = "muon"
    
    c.muon_lr = 0.02
    c.muon_momentum = 0.95
    c.muon_nesterov = True
    c.muon_backend = "newtonschulz5"
    c.muon_auxiliary_lr = lambda: c.muon_lr * 0.1
    
    c.weight_decay = 0.0
    
    c.output_dir = "modernbert-diffusion-muon"
    c.save_strategy = "epoch"
    c.save_total_limit = 1
    c.logging_steps = 200
    
    return c

