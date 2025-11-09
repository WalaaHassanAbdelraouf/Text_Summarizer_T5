"""
Quick demo script to showcase the T5 summarization model
"""

from inference import load_trained_model, generate_summary
import config


def run_demo():
    print("=" * 80)
    print("T5 Text Summarization - Demo")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_trained_model(config.MODEL_SAVE_PATH)
    print("✓ Model loaded successfully\n")
    
    # Example articles for demonstration
    examples = [
        {
            "title": "Tech News",
            "text": "Technology giant Apple announced today the launch of its latest iPhone model featuring advanced AI capabilities and improved battery life. The new device, unveiled at the company's annual conference in California, includes a revolutionary neural engine that processes machine learning tasks up to 50% faster than its predecessor. Industry analysts predict strong sales for the holiday season, with pre-orders beginning next week. The phone will be available in four colors and three storage capacities, with prices starting at $999."
        },
        {
            "title": "Business Update",
            "text": "Global markets experienced significant volatility today as investors reacted to the Federal Reserve's announcement of a surprise interest rate adjustment. The S&P 500 index dropped 2.3% in early trading before recovering somewhat by the closing bell. Economists suggest that the central bank's decision reflects concerns about persistent inflation and its impact on consumer spending. Major tech stocks led the decline, while energy sector shares showed resilience."
        },
        {
            "title": "Scientific Breakthrough",
            "text": "Researchers at Stanford University have made a breakthrough in renewable energy technology by developing a new type of solar panel that converts sunlight to electricity with 40% efficiency, nearly double the rate of conventional panels. The innovation uses a novel arrangement of silicon and perovskite materials that capture a broader spectrum of light. Scientists believe this advancement could significantly accelerate the adoption of solar energy worldwide and help combat climate change."
        }
    ]
    
    # Generate summaries
    print("=" * 80)
    print("Generating Summaries")
    print("=" * 80)
    
    for i, example in enumerate(examples, 1):
        print(f"\n📰 Example {i}: {example['title']}")
        print("-" * 80)
        print(f"Original ({len(example['text'])} chars):")
        print(f"{example['text']}\n")
        
        # Generate summary
        summary = generate_summary(example['text'], model, tokenizer)
        
        print(f"✨ Summary ({len(summary)} chars):")
        print(f"{summary}")
        
        # Calculate compression
        compression = (1 - len(summary) / len(example['text'])) * 100
        print(f"\n📊 Compression: {compression:.1f}%")
        print("=" * 80)
    
    print("\n🎉 Demo completed successfully!")
    print("\nTry your own text:")
    print("  python test.py --mode interactive")


if __name__ == "__main__":
    run_demo()