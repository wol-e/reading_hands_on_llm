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

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")

# Get model outputs
model_output = model.model(input_ids)
lm_head_output = model.lm_head(model_output[0])

# Get logits for the last token
logits = lm_head_output[0, -1]

# Get top 5 token IDs and their probabilities
topk = torch.topk(logits, k=5)
top_ids = topk.indices
top_probs = torch.softmax(topk.values, dim=-1)

# Print top 5 tokens with probabilities
for token_id, prob in zip(top_ids, top_probs):
    print(f"{tokenizer.decode(token_id)}\t{prob.item():.4f}")
