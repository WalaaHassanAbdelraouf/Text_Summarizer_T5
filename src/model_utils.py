"""
Model loading, tokenization, and training utilities
"""

import numpy as np
import evaluate
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import config


def load_model_and_tokenizer():
    print("\n" + "=" * 80)
    print("Loading Model and Tokenizer")
    print("=" * 80)
    
    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    print(f"✓ Loaded {config.MODEL_NAME}")
    
    return model, tokenizer


def preprocess_function(examples, tokenizer):
    # Add task prefix
    inputs = ['summarize: ' + doc for doc in examples['text']]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=config.MAX_INPUT_LENGTH,
        truncation=True
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["headlines"],
            max_length=config.MAX_TARGET_LENGTH,
            truncation=True
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_datasets(train_ds, val_ds, test_ds, tokenizer):
    print("\n" + "=" * 80)
    print("Tokenizing Datasets")
    print("=" * 80)
    
    # Create preprocessing function with tokenizer
    preprocess_fn = lambda examples: preprocess_function(examples, tokenizer)
    
    tokenized_train = train_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=["text", "headlines"]
    )
    
    tokenized_val = val_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=["text", "headlines"]
    )
    
    tokenized_test = test_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=["text", "headlines"]
    )
    
    print("✓ Tokenization complete")
    
    return tokenized_train, tokenized_val, tokenized_test


def setup_metrics(tokenizer):
    """
    Setup evaluation metrics (ROUGE)
    """
    rouge = evaluate.load('rouge')
    
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        
        # Convert logits to token IDs if needed
        if isinstance(preds, tuple):
            preds = preds[0]
        
        if len(preds.shape) == 3:
            preds = np.argmax(preds, axis=-1)
        
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"]
        }
    
    return compute_metrics


def create_training_arguments():
    """
    Create training arguments configuration
    """
    training_args = TrainingArguments(
        output_dir=config.RESULTS_DIR,
        
        # Training parameters
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        
        # Evaluation and saving
        eval_strategy=config.EVAL_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        
        # Logging
        logging_dir=config.LOGS_DIR,
        logging_strategy="steps",
        logging_steps=config.LOGGING_STEPS,
        
        # Optimization
        fp16=config.USE_FP16,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
        optim=config.OPTIMIZER,
        
        # Other settings
        push_to_hub=False,
        report_to="none",
        dataloader_pin_memory=False
    )
    
    return training_args


def create_trainer(model, tokenizer, training_args, 
                   tokenized_train, tokenized_val, compute_metrics):
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    return trainer


def train_model(trainer):
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    train_result = trainer.train()
    
    print("\n✓ Training completed!")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"Training samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
    
    return train_result


def evaluate_model(trainer, tokenized_val, tokenized_test):
    """
    Evaluate model on validation and test sets
    """
    print("\n" + "=" * 80)
    print("Evaluating on Validation Set")
    print("=" * 80)
    
    val_results = trainer.evaluate(tokenized_val)
    
    print("\nValidation Set Results:")
    for key, value in val_results.items():
        print(f"{key}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("Evaluating on Test Set")
    print("=" * 80)
    
    test_results = trainer.evaluate(tokenized_test)
    
    print("\nTest Set Results:")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")
    
    return val_results, test_results


def save_model(model, tokenizer, save_path=config.MODEL_SAVE_PATH):
    print("\n" + "=" * 80)
    print("Saving Model")
    print("=" * 80)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"✓ Model saved to '{save_path}'")