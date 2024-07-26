import transformers
import torch
import os

# Set your Hugging Face token
os.environ["HUGGINGFACE_TOKEN"] = "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"

# Verify the token is set correctly
token = os.getenv("HUGGINGFACE_TOKEN")

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=token)

# Load the model in 8-bit precision
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
    use_auth_token=token
)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your prompt
prompt = "Once upon a time in a land far, far away"

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_length=100, num_return_sequences=1)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)