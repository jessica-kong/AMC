import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


LABEL_TO_IDX = {
    'FA': 'fact',
    'RE': 'reasoning',
    'CI': 'citation',
    'CO': 'comparison',
    'SI': 'simile'
}


@dataclass
class Example:
    text: str
    text_pair: str
    label: str


class MultiLabelDataset(Dataset):
    def __init__(self, all_input_ids, all_labels, pad_token_id):
        self.all_input_ids = all_input_ids
        self.all_labels = all_labels
        self.pad_token_id = pad_token_id

    def __getitem__(self, idx):
        return {
            "input_ids": self.all_input_ids[idx],
            "attention_mask": self.all_input_ids[idx].ne(self.pad_token_id),
            "labels": self.all_labels[idx]
        }

    def __len__(self):
        return len(self.all_input_ids)


def convert_label(label):
    for idx, txt in LABEL_TO_IDX.items():
        label = label.replace(idx, txt)
    return ", ".join(label.split(","))


def prepare_dataset(args, tokenizer: PreTrainedTokenizer, split):
    assert split in ["train", "val", "test"]
    dataset_dir = os.path.join(args.dataset_root, split)

    examples = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".txt"):
            with open(os.path.join(dataset_dir, file), mode="r", encoding="utf-8") as f:
                lines = f.readlines()

            sents = []
            for line in lines:
                spl = line.strip().split('\t')
                sents.append(spl[2])

            for line in lines:
                spl = line.strip().split('\t')
                if len(spl) == 4:
                    s = spl[2]
                    lb = spl[3].split(' ')[1]
                    label_txt = convert_label(lb)
                    support = int(spl[3].split(' ')[0].split(':')[1])
                    supprt_s = sents[support]  # s所支持的句子
                    examples.append(
                        Example(text=s, text_pair=supprt_s, label=label_txt)
                    )

    all_input_ids = []
    all_labels = []
    for example in examples:
        input_ids = tokenizer.encode(
            text=example.text,
            text_pair=example.text_pair,
            padding="max_length",
            max_length=args.max_input_length,
            truncation=True
        )
        labels = tokenizer.encode(
            text=example.label,
            padding="max_length",
            max_length=args.max_target_length,
            truncation=True
        )
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    dataset = MultiLabelDataset(all_input_ids, all_labels, pad_token_id=tokenizer.pad_token_id)

    return dataset, examples
