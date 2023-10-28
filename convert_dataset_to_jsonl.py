import os
import json
from dataclasses import dataclass


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

e2c = {'FA':'事实', 'RE':'道理', 'CI':'引用', 'CO':'对比', 'SI':'比喻'}


def convert_label(label):
    for idx, txt in LABEL_TO_IDX.items():  # LABEL_TO_IDX
        label = label.replace(idx, txt)
    return ", ".join(label.split(","))


def main():
    for split in ["train", "validation", "test"]:
        dataset_dir = os.path.join("data_en", split)

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
                        print(split, file, spl[2])
                        s = spl[2]
                        lb = spl[3].split(' ')[1]
                        label_txt = convert_label(lb)
                        support = int(spl[3].split(' ')[0].split(':')[1])
                        supprt_s = sents[support]  # s所支持的句子
                        # lb = label2vec(lb)  # 变成了有1-2个1的5维向量, e.g. [1, 0, 1, 0, 0]代表FA, CI
                        examples.append(
                            Example(text=s, text_pair=supprt_s, label=label_txt)
                        )

        with open(f"en/{split}.json", mode="w", encoding="utf-8") as f:
            for example in examples:
                prompt = 'argumentative method of'
                hard1 = 'uses the following argumentative method to support'
                hard2 = 'is: '
                data = {
                    "content": f"{example.text}\n{hard1}\n{example.text_pair}",
                    "summary": example.label
                }
                f.write(json.dumps(data))
                f.write("\n")


if __name__ == "__main__":
    main()
