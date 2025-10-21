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
    
    c.optimizer_name = "adamw"
    
    c.learning_rate = 5e-5
    c.weight_decay = 0.01
    c.adam_beta1 = 0.9
    c.adam_beta2 = 0.999
    c.adam_epsilon = 1e-8
    
    c.sgd_momentum = 0.9
    c.sgd_nesterov = True
    
    c.muon_lr = 0.02
    c.muon_momentum = 0.95
    c.muon_nesterov = True
    c.muon_backend = "newtonschulz5"
    c.muon_auxiliary_lr = lambda: c.muon_lr * 0.1
    
    c.output_dir = "modernbert-diffusion-single-with-prefix"
    c.save_strategy = "epoch"
    c.save_total_limit = 1
    c.logging_steps = 200
    
    return c
