import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
from typing import Mapping, List

class TextClassificationDataset(Dataset):

    def __init__(self,
                 texts: List[str],
                 labels: List[str] = None,
                 label_dict: Mapping[str, int] = None,
                 max_seq_length: int = 512,
                 model_name: str = 'distilbert-base-uncased'):

        self.texts = texts
        self.labels = labels
        self.label_dict = label_dict
        self.max_seq_length = max_seq_length

        if self.label_dict is None and labels is not None:
            self.label_dict = dict(zip(sorted(set(labels)),
                                       range(len(set(labels)))))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.sep_vid = self.tokenizer.vocab["[SEP]"]
        self.cls_vid = self.tokenizer.vocab["[CLS]"]
        self.pad_vid = self.tokenizer.vocab["[PAD]"]

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:

        x = self.texts[index]
        x_encoded = self.tokenizer.encode(
            x,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            truncation=True  # 明确启用截断
        ).squeeze(0)

        true_seq_length = x_encoded.size(0)
        pad_size = self.max_seq_length - true_seq_length
        pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()
        x_tensor = torch.cat((x_encoded, pad_ids))

        mask = torch.ones_like(x_encoded, dtype=torch.int8)
        mask_pad = torch.zeros_like(pad_ids, dtype=torch.int8)
        mask = torch.cat((mask, mask_pad))

        output_dict = {
            "features": x_tensor,
            'attention_mask': mask
        }

        if self.labels is not None:
            y = self.labels[index]
            y_encoded = torch.Tensor(
                [self.label_dict.get(y, -1)]
            ).long().squeeze(0)
            output_dict["targets"] = y_encoded

        return output_dict

class DistilBertForSequenceClassification(nn.Module):

    def __init__(self, pretrained_model_name: str, num_classes: int = None):

        super().__init__()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes)

        self.distilbert = AutoModel.from_pretrained(pretrained_model_name,
                                                    config=config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, num_classes)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def forward(self, features, attention_mask=None, head_mask=None):

        outputs = self.distilbert(input_ids=features,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)

        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits