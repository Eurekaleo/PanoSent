import argparse
import logging
import os
import random

import numpy as np
import torch
from accelerate import (
    dispatch_model,
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_in_model,
)
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

LoRA = True
init_empty = False

model_path = "google/flan-t5-xxl"
output_dir = "./lora-flan-t5-xxl"
tokenizer = AutoTokenizer.from_pretrained(model_path)
train_dataset = Dataset.load_from_disk("xxx")
dev_dataset = Dataset.load_from_disk("xxx")

if LoRA:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    if init_empty:
        cuda_list = "0,1".split(",")
        memory = "12GB"
        max_memory = {int(cuda): memory for cuda in cuda_list}
        with init_empty_weights():
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        device_map = infer_auto_device_map(model, max_memory=max_memory)
        load_checkpoint_in_model(model, model_path, device_map=device_map)
        model = dispatch_model(model, device_map=device_map)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    auto_find_batch_size=True,
    # learning_rate=1e-3,
    num_train_epochs=10,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
    report_to="tensorboard",
    predict_with_generate=True,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)
model.config.use_cache = False
trainer.train()

if LoRA:
    model_id = "results/flant5_xxl_trainxepoch-LoRA"
else:
    model_id = "reselts/flant5_xxl_trainxepoch"
trainer.model.save_pretrained(model_id)
tokenizer.save_pretrained(model_id)
