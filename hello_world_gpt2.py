from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Small model that fits easily on your GPU
model_name = "gpt2"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # reduce memory usage
)
model.to(device)

# Test generation
inputs = tokenizer("Hello gpt2, how are you?", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0]))

