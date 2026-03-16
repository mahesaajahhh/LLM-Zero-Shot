import torch
from transformers import pipeline
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    print("Testing pipeline initialization (without loading weights if possible)...")
    # Just check if we can import and setup basic config
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    print("Pipeline initialized successfully on CPU.")
except Exception as e:
    print(f"Error initializing pipeline: {str(e)}")
