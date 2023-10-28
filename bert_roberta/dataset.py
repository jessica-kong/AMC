from dataclasses import dataclass
import os
import logging
import torch
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)
LABEL_TO_IDX = {'FA': 0, 'RE': 1, 'CI': 2, 'CO': 3, 'SI': 4}


@dataclass
class Example:
    text: str
    text_pair: str
    label: str


class MultilabelDataset(Dataset):

    def __init__(self, input_ids, labels):
        super().__init__()
        assert len(input_ids) == len(labels)
        self.input_ids = input_ids
        self.labels = labels

    def __getitem__(self, item):
        return {
            "input_ids": self.input_ids[item],
            "labels": self.labels[item]
        }

    def __len__(self):
        return len(self.input_ids)


def build_dataset(examples, tokenizer, max_length):
    # tokenize and convert to tensors
    all_input_ids = []
    all_labels = []
    for example in examples:
        input_ids = tokenizer.encode(
            text=example.text,
            text_pair=example.text_pair,
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        all_input_ids.append(input_ids)
        all_labels.append(label2vec(example.label))

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_labels = torch.tensor(all_labels, dtype=torch.float)
    dataset = MultilabelDataset(input_ids=all_input_ids, labels=all_labels)

    return dataset


def label2vec(label_txt):
    label = [0 for _ in range(len(LABEL_TO_IDX))]
    for txt in label_txt.split(","):
        if txt in LABEL_TO_IDX:
            label[LABEL_TO_IDX[txt]] = 1
    return label


def one_set(folder, mode):  # 根据mode选择处理train val set下的文件
    folder_path = os.path.join(folder, mode)
    examples = []
    for i, file_name in enumerate(os.listdir(folder_path)):
        txt_path = os.path.join(folder_path, file_name)
        with open(txt_path, 'r', encoding='utf-8') as f:
            sents = []
            for line in f:
                spl = line.strip().split('\t')
                sents.append(spl[2])
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                spl = line.strip().split('\t')
                if len(spl) == 4:
                    s = spl[2]
                    lb = spl[3].split(' ')[1]  # 还没有考虑多标签
                    support = int(spl[3].split(' ')[0].split(':')[1])
                    print(mode, txt_path, len(sents), s, support)
                    supprt_s = sents[support-1]  # s所支持的句子
                    examples.append(
                        Example(
                            text=s,
                            text_pair=supprt_s,
                            label=lb
                        )
                    )

    return examples


def read_data(folder):

    train_examples = one_set(folder, 'test')
    val_examples = one_set(folder, 'val')
    test_examples = one_set(folder, 'test')

    return train_examples, val_examples, test_examples


