from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Load model with 8-bit quantization and automatic device placement
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",          # automatically splits model across GPUs/CPU
    load_in_8bit=True,          # enable 8-bit quantization
    torch_dtype=torch.float16,  # for non-quantized parts
    trust_remote_code=False,
)

# Create a pipeline WITHOUT the device argument
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=50,
    do_sample=False,
)

prompt = "The capital of France is"

# Generate text
output = generator(prompt)
print(output[0]['generated_text'])
