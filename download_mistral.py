from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

# Option 1: Use slow tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
# Option 2: Use LlamaTokenizer explicitly
# from transformers import LlamaTokenizer
# tokenizer = LlamaTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Save
tokenizer.save_pretrained("./pretrained/mistral")
model.save_pretrained("./pretrained/mistral")