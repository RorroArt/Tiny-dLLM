import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_dataset(c):
    dataset = load_dataset(c.dataset_name, c.dataset_config)
    for split in ["train", "validation"]:
        if split in dataset:
            dataset[split] = dataset[split].filter(lambda ex: ex["text"].strip() != "")
    
    tokenizer = AutoTokenizer.from_pretrained(c.model_name)
    tokenizer.model_max_length = c.max_len
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=c.max_len,
            truncation=True,
            padding=False,
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // c.max_len) * c.max_len
        
        result = {
            k: [concatenated[k][i : i + c.max_len] for i in range(0, total_length, c.max_len)]
            for k in concatenated.keys()
        }
        result["attention_mask"] = [[1] * c.max_len for _ in range(len(result["input_ids"]))]
        return result
    
    tokenized = tokenized.map(
        group_texts,
        batched=True,
        remove_columns=tokenized["train"].column_names,
    )
    
    return tokenized, tokenizer


def create_diffusion_collator(tokenizer, c):
    mask_probs = [(i + 1) / c.n_steps for i in range(c.n_steps - 1, -1, -1)]
    
    def diffusion_collator(features):
        batch_input_ids = torch.tensor(
            [f["input_ids"] for f in features], dtype=torch.long
        )
        batch_attention = torch.tensor(
            [f["attention_mask"] for f in features], dtype=torch.long
        )
        
        labels = batch_input_ids.clone()
        
        p = float(mask_probs[torch.randint(low=0, high=len(mask_probs), size=(1,))])
        
        B, L = batch_input_ids.shape
        
        special_ids = set(tokenizer.all_special_ids)
        is_special = torch.zeros_like(batch_input_ids, dtype=torch.bool)
        for sid in special_ids:
            is_special |= batch_input_ids == sid
        
        device = batch_input_ids.device
        pos_idxs = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        is_prefix = pos_idxs < c.prefix_len
        
        mask_candidate = (batch_attention == 1) & (~is_special) & (~is_prefix)
        
        rand = torch.rand_like(batch_input_ids, dtype=torch.float)
        
        mask_positions = (rand < p) & mask_candidate
        
        batch_input_ids[mask_positions] = tokenizer.mask_token_id
        
        labels[~mask_positions] = -100
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": labels,
        }
    
    return diffusion_collator
