# mostly the example from p33 but quantized using chatgpt so it runs on my 6GB of VRAM.

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# Enable 4-bit quantization (saves the most VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",    # NormalFloat4 quantization (good default)
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16"
)

# Load model with quantization + GPU offload
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",            # puts as much as possible on GPU, rest on CPU
    quantization_config=bnb_config,
    trust_remote_code=False,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Create pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

# Prompt
messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]

# Generate
output = generator(messages)
print(output[0]["generated_text"])
