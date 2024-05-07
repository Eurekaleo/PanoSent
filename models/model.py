import json
import os

import numpy as np
import torch
import tqdm
from accelerate import (
    dispatch_model,
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_in_model,
)
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def load_data_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dialogues = []
    step_outputs = []
    for item in data:
        dialogues.append(item["dialogue"])
        step_outputs.append(item["step_outputs"])
    return dialogues, step_outputs


class CoTDataset(Dataset):
    def __init__(self, tokenizer, dialogues, step_outputs, max_length=512):
        self.tokenizer = tokenizer
        self.dialogues = dialogues
        self.step_outputs = step_outputs
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        step_output = self.step_outputs[idx]
        input_text = f"Given the following multi-party dialogue and its accompanying multi-modal data: {dialogue} Identify all discussed targets and their specific aspects mentioned in the dialogue. Extract each target and aspect directly from the text as text spans. Ensure each identified target is paired with its aspect(s), forming target-aspect pairs. Based on the identified target-aspect pairs from the dialogue: {step_output['step1']} For each target-aspect pair, identify the holder and their opinion about it, both directly from the text as text spans. Formulate this information into holder-target-aspect-opinion quadruples, ensuring each element is clearly identified. Considering the holder-target-aspect-opinion quadruples previously identified: {step_output['step2']} For each quadruple, analyze the sentiment polarity associated with the opinion or the context of the dialogue and identify the rationale behind the holder's opinion. The sentiment polarity can only be chosen from ['positive', 'neutral', 'negative']. Based on the identified sextuples from the dialogue: {step_output.get('step3', 'None')} Identify instances where a sentiment flip occurs for the same holder regarding a specific target-aspect pair and determine the trigger type for these flips."
        input_encoding = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        output_encoding = self.tokenizer(
            " ",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": output_encoding["input_ids"].squeeze(0),
        }


dialogues, step_outputs = load_data_from_json("xxxxx")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
dataset = CoTDataset(tokenizer, dialogues, step_outputs)
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    logging_dir="logs",
)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
model_checkpoint_dir = "model_checkpoints"
os.makedirs(model_checkpoint_dir, exist_ok=True)


class SaveModelCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        model.save_pretrained(
            os.path.join(model_checkpoint_dir, f"epoch_{epoch}"), save_weights_only=True
        )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[SaveModelCallback()],
)
trainer.train()

model_name = "google/flan-t5-xxl"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def execute_stage(prompt, model, tokenizer):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=512)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def sequential_reasoning(dialogue):
    prompt1 = f"""Based on the multi-party dialogue and its accompanying multimodal data, please identify all possible targets and their specific aspects mentioned in the dialogue. 
Extract each target and aspect explicitly from the utterance text spans, or infer them implicitly via your understanding of the input data. 
Ensure each identified target is paired with its aspect(s), forming target-aspect pairs."""
    stage1_result = execute_stage(prompt1, model, tokenizer)
    print("\nStage 1 Result:\n", stage1_result)

    prompt2 = f"""Based on the dialogue and each target-aspect pair identified previously, please identify the holder (the person who expresses an opinion, normally should be a speaker of certain dialogue utterance) and the opinion, both either directly extracted from the text or inferred from our understanding of the input data.
Formulate your output into `holder-target-aspect-opinion` quadruples, ensuring each element is clearly identified."""
    stage2_result = execute_stage(prompt2, model, tokenizer)
    print("\nStage 2 Result:\n", stage2_result)

    prompt3 = f"""Based on the dialogue and each holder-target-aspect-opinion quadruple identified previously, please identify the sentiment polarity associated with the opinion and analyze the causal rationale behind it. 
The sentiment polarity should be classified as `positive`, `neutral`, or `negative`. The rationale should be extracted explicitly from the text, or
inferred implicitly via your understanding of the input data.
Formulate your output into `holder-target-aspect-opinion-sentiment-rationale` sextuples, ensuring sentiment polarity is clearly analyzed and the other five elements are clearly identified."""
    stage3_result = execute_stage(prompt3, model, tokenizer)
    print("\nStage 3 Result:\n", stage3_result)

    prompt4 = f"""Based on the dialogue and each `holder-target-aspect-opinion-sentiment-rationale` sextuple, please identify instances where a sentiment flip occurs for the same holder regarding the specific target-aspect pair. 
Determine the trigger type for these flips from the predefined categories: 'introduction of new information', 'logical argumentation', 'participant feedback and interaction', 'personal experience and self-reflection'.
Formulate your output to include the holder, target, aspect, initial sentiment, flipped sentiment, and the trigger type, or state "None" if no flips are identified."""
    stage4_result = execute_stage(prompt4, model, tokenizer)
    print("\nStage 4 Result:\n", stage4_result)

    return stage4_result


class FlanT5Test(object):
    def __init__(self, finetuned_model_path, test_dataset_path):
        self.model_path = finetuned_model_path
        self.test_dataset = Dataset.load_from_disk(test_dataset_path)
        self.test_dataset.set_format(type="torch")
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=32)
        self.tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            finetuned_model_path, device_map="auto"
        )

    def generate_output(self):
        self.model.eval()
        sextuples = []
        for batch in self.test_dataloader:
            input_ids = torch.tensor(batch["input_ids"]).cuda()
            source_mask = torch.tensor(batch["attention_mask"]).cuda()
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=source_mask, do_sample=True, top_k=0
            )
            results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            sextuples.extend([self.parse_sextuple(result) for result in results])
        return sextuples

    def parse_sextuple(self, text):
        parts = text.split(",")
        if len(parts) != 6:
            return ["Unknown"] * 6
        return parts


if __name__ == "__main__":
    base_path = "."
    finetuned_model_path = os.path.join(base_path, "xxxxxx")
    test_dataset_path = os.path.join(base_path, "test")
    test = FlanT5Test(finetuned_model_path, test_dataset_path)
    results = test.generate_output()
    for result in results:
        print(result)
