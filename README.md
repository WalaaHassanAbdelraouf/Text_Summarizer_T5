# T5 Text Summarization Project 📝

A text summarization system using Google's T5-Small transformer model, fine-tuned on news articles to generate concise headlines.

## 🎯 Project Overview

This project implements an end-to-end text summarization pipeline that:
- Preprocesses and cleans news article data
- Fine-tunes the T5-Small model for summarization
- Achieves **43.36% ROUGE-L** score on test data

## 🔬 Model Architecture

- **Base Model**: T5-Small (60M parameters)
- **Task**: Sequence-to-sequence text summarization
- **Input**: News articles (up to 512 tokens)
- **Output**: Concise headlines (up to 64 tokens)

### Training Details

- **Optimizer**: AdamW
- **Learning Rate**: 3e-5 with weight decay
- **Batch Size**: 4 (with gradient accumulation)
- **Mixed Precision**: FP16 training
- **Gradient Checkpointing**: Enabled for memory efficiency
- **Evaluation Metric**: ROUGE (F1 scores)

## 📈 Training Process

1. **Data Preprocessing**: 
   - Remove duplicates and special characters
   - Clean URLs and extra whitespace
   - Sample 7,000 articles for training

2. **Tokenization**: 
   - T5 tokenizer with SentencePiece
   - Task prefix: "summarize: "
   - Dynamic padding with DataCollator

3. **Training**:
   - 3 epochs with early stopping
   - Gradient checkpointing for memory efficiency
   - Automatic evaluation per epoch

4. **Evaluation**:
   - ROUGE-1, ROUGE-2, ROUGE-L metrics
   - Separate validation and test sets

## 🛠️ Advanced Features

### Memory Optimization
- Gradient checkpointing
- FP16 mixed precision training
- Gradient accumulation
- Dynamic batch padding

## 📊 Results

| Metric  | Validation | Test       |
|---------|----------- |------------|
| ROUGE-1 | 45.55%     | **44.21%** |
| ROUGE-2 | 23.92%     | **22.99%** |
| ROUGE-L | 44.47%     | **43.36%** |

