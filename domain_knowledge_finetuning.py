import pandas as pd
import torch
from datasets import Dataset, load_dataset
from random import randrange
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

dataset = load_dataset("Trelis/tiny-shakespeare", split="train")
print (f"Dataset loaded: f{dataset}")
input("Move to next step")

model_id = "NousResearch/Llama-2-7b-chat-hf"
# Get the type
compute_dtype = getattr(torch, "float16")

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)

input("BitsAndBytesConfig initialized: Move to next step")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

input("Tokenizer initialized: Move to next step")
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             device_map="auto")
input("Model loaded: Move to next step")

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)
input("LoraConfig loaded: Move to next step")

args = TrainingArguments(
    output_dir='llama2-7b',
    num_train_epochs=10, # adjust based on the data size
    per_device_train_batch_size=2, # use 4 if you have more GPU RAM
    save_strategy="epoch", #steps
    # evaluation_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    seed=42
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    # eval_dataset=test,
    dataset_text_field='Text',
    peft_config=peft_config,
    max_seq_length=1042,
    tokenizer=tokenizer,
    args=args,
    packing=True,
)
input("TrainingArguments and SFTTrainer loaded: Move to next step")

print("Training Started")
trainer.train()
input("Training Finished: Move to next step")

trainer.save_model()
input("Save Model Locally: Move to next step")

new_model = AutoPeftModelForCausalLM.from_pretrained(
    'llama2-7b',
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Merge LoRA and base model")
# Merge LoRA and base model
merged_model = new_model.merge_and_unload()
input("Merged LoRA and base model: Move to next step")

print("Saving the merged model")
merged_model.save_pretrained("metallama2-7b-tuned-merged", safe_serialization=True)
tokenizer.save_pretrained("metallama2-7b-tuned-merged")
input("Saved the merged model: Move to next step")

print("Pushing the model to HF repo...")
hf_model_repo = "cjsanjay/llama-2-7b-domain-tuned"
merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)
input("Pushed the model to HF repo: Done")

