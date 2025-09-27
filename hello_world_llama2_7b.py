from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Model: LLaMA 2 7B 4-bit quantized
model_name = "meta-llama/Llama-2-7b-hf"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# BitsAndBytes 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",      # automatically place layers on GPU/CPU
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

# Test generation
inputs = tokenizer("Hello LLaMA, how are you?", return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0]))

