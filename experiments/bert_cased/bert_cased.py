# Imports
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np

def preprocess_datasets(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

if __name__ == '__main__':
    # Load dataset
    model_name = "bert-base-cased"
    dataset = load_dataset("stanfordnlp/imdb")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print(train_dataset)
    print(test_dataset)

    # Load tokenizer and model
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Encode splits and remove column "text"

    encoded_train = train_dataset.map(preprocess_datasets, batched=True)
    encoded_test = test_dataset.map(preprocess_datasets, batched=True)
    encoded_train = encoded_train.remove_columns(["text"])
    encoded_test = encoded_test.remove_columns(["text"])

    encoded_train = encoded_train.with_format("torch")
    encoded_test = encoded_test.with_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    training_args = TrainingArguments(
        output_dir="./bert-imdb-baseline",
        eval_strategy="no",  # no eval during training
        save_strategy="no",  # don't save checkpoints each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        # no eval_dataset for now
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # used when we call evaluate()
    )

    trainer.train()

    trainer.save_model("bert-bert_cased")

    test_results = trainer.evaluate(encoded_test)
    print(test_results)