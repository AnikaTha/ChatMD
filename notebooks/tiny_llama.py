#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import torch
import numpy as np


# In[2]:


checkpoint = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# In[3]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print('using', device)


# In[4]:


medqa_train = load_dataset('json', data_files='Data/MedQA/data_clean/questions/US/train.jsonl')
# medqa_train = load_dataset('json', data_files='../Data/MedQA/data_clean/questions/US/train.jsonl')


# In[5]:


from peft import LoraConfig, get_peft_config, get_peft_model, get_peft_model_state_dict, TaskType


# In[6]:


config = LoraConfig(init_lora_weights='gaussian', task_type=TaskType.CAUSAL_LM)


# In[7]:


peft_model = get_peft_model(model, config)


# In[8]:


max_length = 512
def preprocess_function(examples):
    batch_size = len(examples['question'])
    targets = [f"{i}: {answer}" for i,answer in zip(examples['answer_idx'], examples['answer'])]
    raw_inputs = [f"Context: {context} Options: {options} Answer:" for context, options in zip(examples['question'], examples['options'])]
    model_inputs = tokenizer(raw_inputs)
    labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:max_length]
        model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][:max_length]
        labels["input_ids"][i] = labels["input_ids"][i][:max_length]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[9]:


processed_train = medqa_train.map(
    preprocess_function,
    batched=True,
    remove_columns=medqa_train['train'].column_names
)


# In[10]:


processed_train
processed_train['train']


# In[11]:


from transformers import default_data_collator


# In[12]:


import accelerate

epochs = 20
training_args = TrainingArguments(
    output_dir="llama_medqa_clm_lora",
    num_train_epochs=epochs,
    save_total_limit=5,
    per_device_train_batch_size=8,
    warmup_steps=10,
    weight_decay=0.0001,
    dataloader_drop_last=True,
    bf16=True,
    logging_steps=10,
    learning_rate=1e-5,
    # gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    #hub_model_id="smangrul/mistral_lora_clm_with_added_tokens",
    #push_to_hub=True,
    #hub_private_repo=True,
)
training_args.set_save(strategy="steps", steps=3180)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=processed_train['train'],
    data_collator=default_data_collator,
)
# model.config.use_cache = False
trainer.train()


# In[ ]:


trainer.save_model('./models')

