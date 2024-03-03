import os

import torch
import GPUtil
import datasets
import numpy as np
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import roc_curve, auc

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, IntervalStrategy

TT_MODEL_DIR = 't-model/'


class CustomTrainer(Trainer):
    """Loss for unbalanced dataset"""
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 2 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([3.0, 1.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class TransformerTrainer:
    checkpoint = "Geotrend/distilbert-base-ru-cased"

    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.device_type = 'gpu' if len(GPUtil.getAvailable()) > 0 else 'cpu'

    @staticmethod
    def create_ddataset(data_train, data_test):
        # TO-DO test <- val_data.csv
        data_train = data_train.rename(columns={'is_bad': 'label'})
        data_test = data_test.rename(columns={'is_bad': 'label'})
        dd = datasets.DatasetDict(
            {
                "train": Dataset.from_dict(data_train),
                "test":  Dataset.from_dict(data_test),
            }
        )
        return dd

    def tokenize_function(self, batch):
        return self.tokenizer(batch['text_cleaned'], truncation=True, max_length=512, padding="max_length", )

    def encode_ddataset(self, dd):
        dd_encoded = dd.map(self.tokenize_function, batched=True, batch_size=None)
        return dd_encoded

    def train(self, dd_encoded):
        batch_size = 8
        logging_steps = len(dd_encoded["train"]) // batch_size
        model_name = f"{self.checkpoint}-finetuned-a-test"
        training_args = TrainingArguments(
            output_dir=model_name,
            num_train_epochs=2,
            learning_rate=0.5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 3,
            weight_decay=0.01,
            warmup_ratio=0.01,
            disable_tqdm=False,
            logging_steps=logging_steps,
            log_level="error",
            optim='adamw_torch',
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=250,
            save_total_limit=5,
            push_to_hub=False,
            load_best_model_at_end=True,
        )

        trainer = CustomTrainer(  # Trainer CustomTrainer
            model=self.model,
            args=training_args,
            train_dataset=dd_encoded["train"],
            eval_dataset=dd_encoded["test"]
        )

        trainer.train()

        predictions = trainer.predict(dd_encoded["test"])
        fpr, tpr, _ = roc_curve(dd_encoded["test"]['label'], predictions.predictions[:, 1])
        roc_auc = auc(fpr, tpr)
        print(f'ROC_AUC for Transformer: {roc_auc}')

        trainer.save_model('t-model', '.')

    def prepare_validation(self, data_val):
        # TO-DO объединить с create_dataset
        data_val = data_val.rename(columns={'is_bad': 'label'})
        dd_val = datasets.DatasetDict(
            {
                "validation":  Dataset.from_dict(data_val),
            }
        )

        dd_val_encoded = dd_val.map(self.tokenize_function, batched=True, batch_size=None)
        return dd_val_encoded

    @staticmethod
    def load_model():
        model_loaded = AutoModelForSequenceClassification.from_pretrained(f'{os.getenv("HOME")}/{TT_MODEL_DIR}')
        return Trainer(model=model_loaded)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
