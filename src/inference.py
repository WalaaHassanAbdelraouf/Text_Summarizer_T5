"""
Inference and summary generation utilities
"""

import random
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import config


def load_trained_model(model_path=config.MODEL_SAVE_PATH):
    print(f"Loading model from {model_path}...")
    
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    
    return model, tokenizer


def generate_summary(text, model, tokenizer, max_length=None, num_beams=None):
    if max_length is None:
        max_length = config.MAX_TARGET_LENGTH
    if num_beams is None:
        num_beams = config.NUM_BEAMS
    
    # Prepare input
    input_text = "summarize: " + text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=config.MAX_INPUT_LENGTH,
        truncation=True
    )
    
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=config.LENGTH_PENALTY,
            early_stopping=config.EARLY_STOPPING,
            no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE
        )
    
    # Decode summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary


def test_on_samples(test_df, model, tokenizer, num_samples=5):
    print("\n" + "=" * 80)
    print("Testing on Sample Articles")
    print("=" * 80)
    
    # Select random samples
    sample_indices = random.sample(range(len(test_df)), num_samples)
    
    for i, idx in enumerate(sample_indices, 1):
        sample = test_df.iloc[idx]
        generated = generate_summary(sample['text'], model, tokenizer)
        
        print(f"\n{'=' * 80}")
        print(f"Sample {i}/{num_samples}")
        print(f"{'=' * 80}")
        print(f"📄 Original Text ({len(sample['text'])} chars):")
        print(f"{sample['text'][:300]}...")
        print(f"\n🎯 Actual Headline: {sample['headlines']}")
        print(f"🤖 Generated Summary: {generated}")


def interactive_summarization(model, tokenizer):
    print("\n" + "=" * 80)
    print("Interactive Summarization Mode")
    print("=" * 80)
    print("Enter text to summarize (or 'quit' to exit)\n")
    
    while True:
        text = input("Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        summary = generate_summary(text, model, tokenizer)
        print(f"\n✨ Summary: {summary}\n")