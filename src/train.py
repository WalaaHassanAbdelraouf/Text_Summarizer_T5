"""
Main training script for T5 Text Summarization
"""

import torch
import gc
from data_utils import load_and_prepare_data, split_data, create_datasets
from model_utils import (
    load_model_and_tokenizer,
    tokenize_datasets,
    setup_metrics,
    create_training_arguments,
    create_trainer,
    train_model,
    evaluate_model,
    save_model
)


def main():
    print("=" * 80)
    print("T5 Text Summarization - Training Pipeline")
    print("=" * 80)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data()
    
    # Step 2: Split data
    train_df, val_df, test_df = split_data(df)
    
    # Step 3: Create datasets
    train_ds, val_ds, test_ds = create_datasets(train_df, val_df, test_df)
    
    # Step 4: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Step 5: Tokenize datasets
    tokenized_train, tokenized_val, tokenized_test = tokenize_datasets(
        train_ds, val_ds, test_ds, tokenizer
    )
    
    # Step 6: Setup metrics
    compute_metrics = setup_metrics(tokenizer)
    
    # Step 7: Create training arguments
    training_args = create_training_arguments()
    
    # Step 8: Create trainer
    trainer = create_trainer(
        model, tokenizer, training_args,
        tokenized_train, tokenized_val, compute_metrics
    )
    
    # Step 9: Train model
    train_result = train_model(trainer)
    
    # Step 10: Evaluate model
    val_results, test_results = evaluate_model(
        trainer, tokenized_val, tokenized_test
    )
    
    # Step 11: Save model
    save_model(model, tokenizer)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n" + "=" * 80)
    print("Training Pipeline Complete!")
    print("=" * 80)
    print(f"\nFinal Test Results:")
    print(f"  ROUGE-1: {test_results['eval_rouge1']:.4f}")
    print(f"  ROUGE-2: {test_results['eval_rouge2']:.4f}")
    print(f"  ROUGE-L: {test_results['eval_rougeL']:.4f}")
    

if __name__ == "__main__":
    main()