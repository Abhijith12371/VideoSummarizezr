#!/usr/bin/env python3
# Complete Controllable Text Summarizer using CtrlSum
# Supports: Default, Query-based, Length-controlled, Domain-specific, and Extractive/Abstractive summaries

import torch
from summarizers import Summarizers

def check_gpu():
    """Check if GPU is available for faster processing"""
    if torch.cuda.is_available():
        device = "cuda"
        print("✅ Using GPU for faster summarization")
    else:
        device = "cpu"
        print("⚠️ GPU not available, using CPU (slower)")
    return device

def initialize_summarizer(device):
    """Initialize the summarizer with specified device"""
    return Summarizers(device=device)

def sample_text():
    with open("transcript.txt","r") as f:
        content=f.read()
    """Return sample text for demonstration"""
    return content

def generate_summaries(summarizer, text):
    """Generate all types of summaries"""
    results = {}
    
    # A. Default Summary
    results['default'] = summarizer.summarize(text)
    
    # B. Query-Based Summary
    results['query_healthcare'] = summarizer.summarize(text, query="How is AI used in healthcare?")
    results['query_finance'] = summarizer.summarize(text, query="How is AI used in finance?")
    
    # C. Length-Controlled Summary
    results['short'] = summarizer.summarize(text, max_length=50)
    results['medium'] = summarizer.summarize(text, max_length=100)
    results['long'] = summarizer.summarize(text, max_length=150)
    
    # D. Domain-Specific Summaries
    results['research'] = summarizer.summarize(text, mode="paper")
    results['news'] = summarizer.summarize(text, mode="news")
    
    # E. Extractive vs. Abstractive
    results['extractive'] = summarizer.summarize(text, style="extractive")
    results['abstractive'] = summarizer.summarize(text, style="abstractive")
    
    # F. Controlled Summary with Prompt
    results['controlled_ethics'] = summarizer.summarize(
        text, 
        control="Must discuss ethical concerns of AI"
    )
    
    return results

def display_results(results):
    """Display all generated summaries"""
    print("\n" + "="*50 + "\n📝 SUMMARY RESULTS\n" + "="*50)
    
    print("\n🔹 DEFAULT SUMMARY:")
    print(results['default'])
    
    print("\n🔍 QUERY-BASED SUMMARIES:")
    print("\nHealthcare Focus:")
    print(results['query_healthcare'])
    print("\nFinance Focus:")
    print(results['query_finance'])
    
    print("\n✂️ LENGTH-CONTROLLED SUMMARIES:")
    print("\nShort (50 chars):")
    print(results['short'])
    print("\nMedium (100 chars):")
    print(results['medium'])
    print("\nLong (150 chars):")
    print(results['long'])
    
    print("\n📚 DOMAIN-SPECIFIC SUMMARIES:")
    print("\nResearch Style:")
    print(results['research'])
    print("\nNews Style:")
    print(results['news'])
    
    print("\n🔄 EXTRACTIVE VS ABSTRACTIVE:")
    print("\nExtractive (Key Sentences):")
    print(results['extractive'])
    print("\nAbstractive (Paraphrased):")
    print(results['abstractive'])
    
    print("\n🎯 CONTROLLED SUMMARY (ETHICS FOCUS):")
    print(results['controlled_ethics'])

def save_to_file(results, filename="summary_results.txt"):
    """Save all summaries to a text file"""
    with open(filename, "w") as f:
        f.write("="*50 + "\n📝 SUMMARY RESULTS\n" + "="*50 + "\n")
        
        f.write("\n🔹 DEFAULT SUMMARY:\n")
        f.write(results['default'] + "\n")
        
        f.write("\n🔍 QUERY-BASED SUMMARIES:\n")
        f.write("\nHealthcare Focus:\n")
        f.write(results['query_healthcare'] + "\n")
        f.write("\nFinance Focus:\n")
        f.write(results['query_finance'] + "\n")
        
        f.write("\n✂️ LENGTH-CONTROLLED SUMMARIES:\n")
        f.write("\nShort (50 chars):\n")
        f.write(results['short'] + "\n")
        f.write("\nMedium (100 chars):\n")
        f.write(results['medium'] + "\n")
        f.write("\nLong (150 chars):\n")
        f.write(results['long'] + "\n")
        
        f.write("\n📚 DOMAIN-SPECIFIC SUMMARIES:\n")
        f.write("\nResearch Style:\n")
        f.write(results['research'] + "\n")
        f.write("\nNews Style:\n")
        f.write(results['news'] + "\n")
        
        f.write("\n🔄 EXTRACTIVE VS ABSTRACTIVE:\n")
        f.write("\nExtractive (Key Sentences):\n")
        f.write(results['extractive'] + "\n")
        f.write("\nAbstractive (Paraphrased):\n")
        f.write(results['abstractive'] + "\n")
        
        f.write("\n🎯 CONTROLLED SUMMARY (ETHICS FOCUS):\n")
        f.write(results['controlled_ethics'] + "\n")
    
    print(f"\n📂 All summaries saved to '{filename}'")

def main():
    # Step 1: Setup environment
    device = check_gpu()
    
    # Step 2: Initialize summarizer
    summarizer = initialize_summarizer(device)
    
    # Step 3: Get sample text
    text = sample_text()
    
    # Step 4: Generate all summaries
    print("\n⏳ Generating summaries...")
    results = generate_summaries(summarizer, text)
    
    # Step 5: Display results
    display_results(results)
    
    # Step 6: Save to file
    save_to_file(results)

if __name__ == "__main__":
    main()