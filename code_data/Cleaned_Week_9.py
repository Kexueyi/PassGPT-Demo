#!/usr/bin/env python
# coding: utf-8

# # Transformer LLMs

# ## Fine-tuning a pretrained BERT Model

# In the first section of this notebook, we will focus on fine-tuning a pretrained BERT model for our use.

# We first load the yelp reviews dataset:



from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
dataset["train"][100]


# We now use a tokenizer and the map function contained within 'datasets' to preprocess the dataset.



from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


# We now split out a smaller training dataset to fine tune our model quickly



small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


# Transformers provides a Trainer class optimized for training Transformers models, making it easier to start training without manually writing your own training loop. The Trainer API supports a wide range of training options and features such as logging, gradient accumulation, and mixed precision.



from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)


# We next create a TrainingArguments class that contains all hyperparameters that are available for tuning and flags for activating our various training options. We will start with the default training parameters.



from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")


# As trainer does not evaluate model performance during training, we need to create a function for trainer to compute and report metrics.



import numpy as np
import evaluate

metric = evaluate.load("accuracy")


# We can call compute on metric to evaluate the perfomance of our model.



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# The model requires a instance of the trainer class in order to train it.



from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", 
                                  evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)




trainer.train()


# ## Fine-tuning a RoBERTa model



import torch
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from huggingface_hub import HfFolder, notebook_login




model_id = "roberta-base"
dataset_id = "ag_news"




from datasets import load_dataset
# Load dataset
dataset = load_dataset(dataset_id)

# Training and testing datasets
train_dataset = dataset['train']
test_dataset = dataset["test"].shard(num_shards=2, index=0)

# Validation dataset
val_dataset = dataset['test'].shard(num_shards=2, index=1)

# Preprocessing
tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

# This function tokenizes the input text using the RoBERTa tokenizer. 
# It applies padding and truncation to ensure that all sequences have the same length (256 tokens).
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))




# Set dataset format
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])




# We will need this to directly output the class names when using the pipeline without mapping the labels later.
# Extract the number of classes and their names
num_labels = dataset['train'].features['label'].num_classes
class_names = dataset["train"].features["label"].names
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

# Create an id2label mapping
id2label = {i: label for i, label in enumerate(class_names)}

# Update the model's configuration with the id2label mapping
config = AutoConfig.from_pretrained(model_id)
config.update({"id2label": id2label})




# TrainingArguments
training_args = TrainingArguments(
    output_dir="test_trainer",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,                     
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
)

# Model
model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


# Note: The following chunk of code will take hours to run, so it is not run here.



# Fine-tune the model
trainer.train()




# Evaluate the model
trainer.evaluate()






