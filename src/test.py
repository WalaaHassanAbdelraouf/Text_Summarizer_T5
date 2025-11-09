"""
Testing script for trained T5 model
"""

import argparse
from data_utils import load_and_prepare_data, split_data
from inference import load_trained_model, test_on_samples, interactive_summarization


def main():
    """
    Main testing pipeline
    """
    parser = argparse.ArgumentParser(description='Test T5 Summarization Model')
    parser.add_argument(
        '--mode',
        type=str,
        default='samples',
        choices=['samples', 'interactive'],
        help='Testing mode: samples or interactive'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to test (for samples mode)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./models',
        help='Path to trained model'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("T5 Text Summarization - Testing")
    print("=" * 80)
    
    # Load trained model
    model, tokenizer = load_trained_model(args.model_path)
    
    if args.mode == 'samples':
        # Load test data
        print("\nLoading test data...")
        df = load_and_prepare_data()
        _, _, test_df = split_data(df)
        
        # Test on samples
        test_on_samples(test_df, model, tokenizer, args.num_samples)
        
    elif args.mode == 'interactive':
        # Interactive mode
        interactive_summarization(model, tokenizer)
    
    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()