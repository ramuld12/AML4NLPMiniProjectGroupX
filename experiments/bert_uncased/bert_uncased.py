#%% md
# # Setup
#%%
# Imports
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np

#%%
import sys; print(sys.executable)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    print("CUDA device name:", torch.cuda.get_device_name(0))
#%%
# Load dataset
dataset = load_dataset("stanfordnlp/imdb")

train_validation_dataset = dataset["train"].train_test_split(test_size=0.1)  
train_dataset = train_validation_dataset["train"]
validation_dataset = train_validation_dataset["test"]
test_dataset = dataset["test"]          

print("Train size:", len(train_dataset))
print("Validation size:", len(validation_dataset))
print("Test size:", len(test_dataset))
#%%
# Load tokenizer and model
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(
    model_name
)

#%%
def preprocess_datasets(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256
    )
#%%
# Encode splits and remove column "text"
encoded_train = train_dataset.map(preprocess_datasets, batched=True)
encoded_validation = validation_dataset.map(preprocess_datasets, batched=True)
encoded_test = test_dataset.map(preprocess_datasets, batched=True)
#%%
encoded_train = encoded_train.remove_columns(["text"])
encoded_validation = encoded_validation.remove_columns(["text"])
encoded_test = encoded_test.remove_columns(["text"])
#%%
encoded_train = encoded_train.with_format("torch")
encoded_validation = encoded_validation.with_format("torch")
encoded_test = encoded_test.with_format("torch")
#%%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
        "precision": precision.compute(predictions=preds, references=labels)["precision"],
        "recall": recall.compute(predictions=preds, references=labels)["recall"]
    }

#%%


#%%
import wandb
import datatime
for run in range(5):
    print(f"Starting run {run+1}")
    training_args = TrainingArguments(
        output_dir=f"./bert_uncased_output{run+1}",
        eval_strategy="epoch",  
        save_strategy="epoch",           
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_steps=100,
        logging_first_step=True,
        load_best_model_at_end=True,
        report_to="wandb", #set to 'none' if no report
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    now = datetime.datetime.now()
    run = wandb.init(
        project="AML4NLPMiniProjectGroupX",    # Your project name (shows in wandb dashboard)
        name=f"{model_name}:{now.timer()}",       # Run name
        config={
            "model": model_name,
            "dataset": "IMDb",
            "task": "Sentiment Analysis",
            "train_size": len(train_dataset),
            "val_size": len(validation_dataset),
            "test_size": len(test_dataset),
            "learning_rate": training_args.learning_rate,
            "batch_train": training_args.per_device_train_batch_size,
            "batch_eval": training_args.per_device_eval_batch_size,
            "epochs": training_args.num_train_epochs,
            "weight_decay": training_args.weight_decay,
            "optimizer": "AdamW",
            "scheduler": "linear",   # Trainer default
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train, 
        eval_dataset=encoded_validation,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    run.finish()
#%%

#%%
trainer.save_model("bert_uncased_model")
#%%
trainer.evaluate(encoded_test)
#%%
 
#%%
