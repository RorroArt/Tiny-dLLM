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
    
    c.learning_rate = 3e-4
    c.weight_decay = 0.01
    c.adam_beta1 = 0.9
    c.adam_beta2 = 0.999
    c.adam_epsilon = 1e-8
    
    c.sgd_momentum = 0.9
    c.sgd_nesterov = True
    
    c.max_grad_norm = 1.0
    
    c.wandb_project = "tiny-dllm"
    c.wandb_run_name = "test"
    c.use_wandb = True
    
    c.output_dir = "modernbert-diffusion-single-with-prefix"
    c.save_strategy = "epoch"
    c.save_total_limit = 1
    c.logging_steps = 10
    
    return c
