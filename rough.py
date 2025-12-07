# Install necessary libraries
# !pip install ollama pandas torch

import ollama
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the LLaMA3 model via Ollama
model_name = "llama3.2:1b"

# Load dataset (assumed CSV file with 'tweet' and 'emoji' columns)
data = pd.read_csv(r"C:\Users\a\Desktop\LLAMA 3.2\twitter.csv")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data["tweet"], data["emoji"], test_size=0.025, random_state=42)

# Prepare prompt template for training
def format_prompt(tweet, emoji):
    return f"### Instruction: Given a tweet, predict the associated emoji.\n### Tweet: {tweet}\n### Emoji: {emoji}\n"

train_data = "\n".join([format_prompt(tweet, emoji) for tweet, emoji in zip(x_test, y_test)])

# Save training data to a file
train_file = "training_data_1.txt"
with open(train_file, "w", encoding="utf-8") as f:
    f.write(train_data)

# Fine-tune the model using Ollama (requires CLI-based fine-tuning)
ollama.pull(model_name)  # Ensure model is available

# Specify where the model will be saved
model_save_path = r"C:\Users\a\Desktop\LLAMA 3.2\fine_tuned_llama3.2:1b"

print("Fine-tuning must be done via the Ollama CLI. Run the following command:")
print(f"ollama create llama3-finetuned -f Modelfile")
print(f"Once the fine-tuning is complete, the model will be saved in: {model_save_path}")
