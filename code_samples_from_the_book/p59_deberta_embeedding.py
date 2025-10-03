from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

tokens = tokenizer("Hello World", return_tensors='pt')

output = model(**tokens)[0]

print(output.shape)

for token in tokens["input_ids"][0]:
    print(tokenizer.decode(token))

print(output)