import gc
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import RobertaTokenizerFast, TrainingArguments, Trainer, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from preprocess_interviews import get_token_labels
from transformers import DataCollatorForTokenClassification

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class RobertaForMultiLabelTokenClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            target: torch.LongTensor = labels.view(logits.size())
            loss = loss_fct(logits, target.float())

        logits = torch.sigmoid(logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def get_compute_metrics(label_list):
    threshold = 0.5

    def compute_metrics(p):
        predictions: list[list[list[int]]] = p[0]
        labels: list[list[list[int]]] = p[1]

        # TODO Remove ignored index (special tokens)
        # These tokens will reduce accuracy
        true_labels = [[
            el > threshold for el in item
        ] for sublist in labels for item in sublist if item[0] > -50]
        pred_labels = [[
            el > threshold for el in p
        ] for true, pred in zip(labels, predictions) for t, p in zip(true, pred) if t[0] > -50]

        true_labels = [item for sublist in true_labels for item in sublist]
        pred_labels = [item for sublist in pred_labels for item in sublist]

        micro_f1_value = f1_score(true_labels, pred_labels, average='micro')
        macro_f1_value = f1_score(true_labels, pred_labels, average='macro')
        micro_precision_value = precision_score(true_labels, pred_labels, average='micro')
        macro_precision_value = precision_score(true_labels, pred_labels, average='macro')
        micro_recall_value = recall_score(true_labels, pred_labels, average='micro')
        macro_recall_value = recall_score(true_labels, pred_labels, average='macro')
        accuracy_value = accuracy_score(true_labels, pred_labels)

        return {
            "micro_f1": micro_f1_value,
            "macro_f1": macro_f1_value,
            "micro_precision": micro_precision_value,
            "macro_precision": macro_precision_value,
            "micro_recall": micro_recall_value,
            "macro_recall": macro_recall_value,
            'accuracy': accuracy_value
        }

    return compute_metrics


# testing code
if __name__ == "__main__":
    batch_size = 2

    # cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # print start info
    print("starting to train BERT with INDEXQUAL data")
    print("preprocessing ...")

    data, unique_labels = get_token_labels(code_mode="sentiment")
    df = pd.DataFrame(data)

    print(unique_labels)
    label2id = {k: v for v, k in enumerate(unique_labels)}
    print("label2id", label2id)
    id2label = {v: k for v, k in enumerate(unique_labels)}
    print("id2label", id2label)

    model_checkpoint = 'pdelobelle/robbert-v2-dutch-base'
    tokenizer = RobertaTokenizerFast.from_pretrained(model_checkpoint)
    model = RobertaForMultiLabelTokenClassification.from_pretrained(model_checkpoint,
                                                                    num_labels=len(unique_labels),
                                                                    classifier_dropout=0.15,
                                                                    id2label=id2label,
                                                                    label2id=label2id)

    splitter = GroupShuffleSplit(test_size=.10, n_splits=2, random_state=7)
    split = splitter.split(df, groups=df['transcript_name'])
    train_ids, test_ids = next(split)

    train = df.iloc[train_ids]
    test = df.iloc[test_ids]

    print(len(set(train['transcript_name'])))
    print(set(train['transcript_name']))
    print(len(set(test['transcript_name'])))
    print(set(test['transcript_name']))

    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)
    tokenized_datasets = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    print(tokenized_datasets)

    model_name = model_checkpoint.split("/")[-1]
    project_name = f"{model_name}-finetuned-indexqual-sentiment"
    args = TrainingArguments(
        project_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        save_total_limit=2,
        weight_decay=0.01
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    compute_metrics_fun = get_compute_metrics(unique_labels)

    wandb.login()
    wandb.init(project=project_name)

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fun
    )

    trainer.train()

    wandb.finish()

    
