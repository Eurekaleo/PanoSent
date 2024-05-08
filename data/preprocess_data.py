import json
import os

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

# 数据集目录
data_directory = "xxxx"

tokenizer_name = "google/flan-t5-xxl"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


def preprocess_function(sample, padding="max_length"):
    model_inputs = tokenizer(
        sample["data"], max_length=max_source_length, padding=padding, truncation=True
    )

    labels = tokenizer(
        text_target=sample["gold"],
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )

    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


dataset = {}
type_list = ["train", "val", "test"]
for type in type_list:
    data_path = data_directory + type + ".json"
    with open(data_path, "r") as f:
        dataset[type] = json.load(f)
        dataset[type] = Dataset.from_dict(dataset[type])

tokenized_inputs = concatenate_datasets(
    [dataset["train"], dataset["val"], dataset["test"]]
).map(
    lambda x: tokenizer(x["data"], truncation=True),
    batched=True,
    remove_columns=["data", "gold"],
)
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
max_source_length = int(np.max(input_lenghts))
print(f"Max source length: {max_source_length}")

tokenized_targets = concatenate_datasets(
    [dataset["train"], dataset["val"], dataset["test"]]
).map(
    lambda x: tokenizer(x["gold"], truncation=True),
    batched=True,
    remove_columns=["data", "gold"],
)
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
max_target_length = int(np.max(target_lenghts))
print(f"Max source length: {max_target_length}")

tokenized_dataset = {}
tokenized_dataset["train"] = dataset["train"].map(
    preprocess_function, batched=True, remove_columns=["data", "gold"]
)
tokenized_dataset["val"] = dataset["val"].map(
    preprocess_function, batched=True, remove_columns=["data", "gold"]
)
tokenized_dataset["test"] = dataset["test"].map(
    preprocess_function, batched=True, remove_columns=["data", "gold"]
)
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")


# 保存数据集
tokenized_dataset["train"].save_to_disk(os.path.join(data_directory, "xxx"))
tokenized_dataset["val"].save_to_disk(os.path.join(data_directory, "xxx"))
tokenized_dataset["test"].save_to_disk(os.path.join(data_directory, "xxx"))
