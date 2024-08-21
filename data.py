import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME =  r"C:\Users\khade\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers"
OUTPUT_FILE = "synthetic_data.txt"
PROMPT = "Generate synthetic data for detection of malware and running high memory apps on background:"
NUM_SAMPLES = 100  # Number of synthetic data samples to generate
MAX_LENGTH = 512  # Maximum length of the generated text

# Load the model and tokenizer
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def generate_synthetic_data(tokenizer, model, prompt, num_samples, max_length):
    synthetic_data = []
    inputs = tokenizer(prompt, return_tensors = "pt")
    
    for _ in range(num_samples):
        outputs = model.generate(
            inputs["input_ids"],
            max_length = max_length,
            num_return_sequences = 1,
            do_sample = True,
            top_p = 0.95,
            temperature = 0.7
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens = True)
        synthetic_data.append(generated_text)
    
    return synthetic_data

def save_data_to_file(data, filename):
    with open(filename, 'w') as file:
        for line in data:
            file.write(line + '\n')

def main():
    tokenizer, model = load_model_and_tokenizer()
    data = generate_synthetic_data(tokenizer, model, PROMPT, NUM_SAMPLES, MAX_LENGTH)
    save_data_to_file(data, OUTPUT_FILE)
    print(f"Generated data has been saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
