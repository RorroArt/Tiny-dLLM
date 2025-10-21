import sws
import wandb
from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments

from .data import prepare_dataset, create_diffusion_collator
from .optimizer import create_optimizer


def train(c):
    if c.use_wandb:
        wandb.init(
            project=c.wandb_project,
            name=c.wandb_run_name,
            config={
                "model_name": c.model_name,
                "dataset_name": c.dataset_name,
                "dataset_config": c.dataset_config,
                "learning_rate": c.learning_rate,
                "weight_decay": c.weight_decay,
                "batch_size": c.batch_size,
                "num_epochs": c.num_epochs,
                "max_len": c.max_len,
                "prefix_len": c.prefix_len,
                "n_steps": c.n_steps,
                "optimizer_name": c.optimizer_name,
                "max_grad_norm": c.max_grad_norm,
            }
        )
    
    print(f"Loading dataset: {c.dataset_name}/{c.dataset_config}")
    tokenized, tokenizer = prepare_dataset(c)
    
    print(f"Loading model: {c.model_name}")
    model = AutoModelForMaskedLM.from_pretrained(c.model_name)
    
    print("Creating diffusion collator...")
    diffusion_collator = create_diffusion_collator(tokenizer, c)
    
    if c.optimizer_name == "adamw":
        training_args = TrainingArguments(
            output_dir=c.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=c.num_epochs,
            per_device_train_batch_size=c.batch_size,
            save_strategy=c.save_strategy,
            save_total_limit=c.save_total_limit,
            logging_steps=c.logging_steps,
            learning_rate=c.learning_rate,
            weight_decay=c.weight_decay,
            adam_beta1=c.adam_beta1,
            adam_beta2=c.adam_beta2,
            adam_epsilon=c.adam_epsilon,
            optim="adamw_torch",
            max_grad_norm=c.max_grad_norm,
            report_to="wandb" if c.use_wandb else "none",
            run_name=c.wandb_run_name,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=diffusion_collator,
            tokenizer=tokenizer,
        )
    else:
        print(f"Creating custom optimizer: {c.optimizer_name}")
        optimizer = create_optimizer(model, c)
        
        training_args = TrainingArguments(
            output_dir=c.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=c.num_epochs,
            per_device_train_batch_size=c.batch_size,
            save_strategy=c.save_strategy,
            save_total_limit=c.save_total_limit,
            logging_steps=c.logging_steps,
            max_grad_norm=c.max_grad_norm,
            report_to="wandb" if c.use_wandb else "none",
            run_name=c.wandb_run_name,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=diffusion_collator,
            tokenizer=tokenizer,
            optimizers=(optimizer, None),
        )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {c.output_dir}")
    trainer.save_model(c.output_dir)
    tokenizer.save_pretrained(c.output_dir)
    
    if c.use_wandb:
        wandb.finish()
    
    print("Finished diffusion-style finetuning with prefix tokens never masked\n")


if __name__ == "__main__":
    sws.run(train)
