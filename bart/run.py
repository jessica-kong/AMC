import json
import logging
import numpy as np
import os
from tqdm import tqdm
import math
import pandas as pd
import random
import sys
import argparse
import time
from prettytable import PrettyTable

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoModelForSeq2SeqLM, AutoTokenizer

from utils import (
    model_summary,
    EarlyStopController,
    MixedPrecisionManager,
    compute_acc_for_text,
    save_eval_details,
    f1_scores
)
from data import prepare_dataset


path = '../data_en'


def train(args, model, tokenizer):
    logging.info("***** Loading Datasets *****")
    train_dataset, train_examples = prepare_dataset(
        args=args, tokenizer=tokenizer, split="train",
    )
    logging.info(f"Train dataset is prepared, size: {len(train_dataset)}")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    valid_dataset, valid_examples = prepare_dataset(
        args=args, tokenizer=tokenizer, split="val",
    )
    logging.info(f"Validation dataset is prepared, size: {len(valid_dataset)}")
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    logging.info("***** Preparing Training Utils *****")
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate
    )

    # calculate max steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    if args.warmup_steps >= 1:
        args.warmup_steps = int(args.warmup_steps)
    elif args.warmup_steps >= 0:
        args.warmup_steps = int(args.warmup_steps * args.max_train_steps)
    else:
        raise ValueError(f"Invalid warmup steps: {args.warmup_steps}")

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # mixed precision
    amp = MixedPrecisionManager(activated=args.fp16)

    # early stop
    best_model_path = os.path.join(
        args.model_dir, f"best_acc", "model.pt"
    )
    early_stop = EarlyStopController(
        patience=args.patience,
        best_model_path=best_model_path,
        higher_is_better=True,
    )

    # batch size per device, total batch size
    if args.num_devices > 1:
        batch_size_per_device = args.batch_size // args.num_devices
        if batch_size_per_device * args.num_devices != args.batch_size:
            raise ValueError(
                f"The total batch size {args.batch_size=} is not an integer multiple "
                f"of the device count: {args.num_devices}"
            )
    else:
        batch_size_per_device = args.batch_size
    total_batch_size = args.batch_size * args.gradient_accumulation_steps

    logging.info("***** Training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {args.num_epochs}")
    logging.info(f"  Batch size per device = {batch_size_per_device}")
    logging.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
    logging.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {args.max_train_steps}")

    completed_steps = 0
    model.zero_grad()

    train_bar = tqdm(
        range(args.max_train_steps),
        total=args.max_train_steps,
        ascii=True,
    )
    for epoch in range(args.num_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            with amp.context():
                # places all tensors in the dict to the right device
                for k, v in batch.items():
                    batch[k] = v.to(args.device)

                # passes batch through model
                outputs = model(**batch)
                loss = outputs.loss

                # gets mean loss
                if args.num_devices > 1:
                    loss = loss.mean()

                # normalizes loss
                loss = loss / args.gradient_accumulation_steps

            # backwards loss by amp manager
            amp.backward(loss)

            if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
            ):
                # amp.step() includes gradient clipping and optimizer.step()
                amp.step(
                    model=model,
                    optimizer=optimizer,
                    max_grad_norm=args.max_grad_norm,
                )
                lr_scheduler.step()
                completed_steps += 1
                train_bar.update(1)

            if completed_steps % args.logging_steps == 0:
                logging.info(
                    {
                        "global_step": completed_steps,
                        "epoch": completed_steps / num_update_steps_per_epoch,
                        "loss": loss.item(),
                    }
                )

            if completed_steps >= args.max_train_steps:
                break

        logging.info("Start validation")
        valid_results = __eval(
            args=args,
            model=model,
            dataloader=valid_dataloader,
            tokenizer=tokenizer,
            examples=valid_examples,
            split="val",
            epoch=epoch,
        )

        early_stop(
            score=valid_results[f"val/acc"],
            model=model,
            epoch=epoch,
            step=completed_steps,
        )
        if not early_stop.hit:
            logging.info(
                f"Early stopping counter: {early_stop.counter}/{early_stop.patience}"
            )

        if early_stop.early_stop:
            logging.info(f"Early stopping is triggered")
            break

    logging.info("End of training")

    # load best model at end of training
    model = early_stop.load_best_model()

    return model


def test(args, model, tokenizer):
    logging.info("***** Testing *****")
    test_dataset, test_examples = prepare_dataset(
        args=args,
        tokenizer=tokenizer,
        split="test",
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    logging.info(f"Test dataset is prepared, size: {len(test_dataset)}")

    model.to(args.device)
    if args.num_devices > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    test_results = __eval(
        args=args,
        model=model,
        dataloader=test_dataloader,
        tokenizer=tokenizer,
        examples=test_examples,
        split="test",
    )
    logging.info("End of testing")

    return test_results


def __eval(args, model, dataloader, tokenizer, examples, split, epoch=None):
    assert split in ["val", "test"]
    assert split == "test" or epoch is not None

    # statistics
    num_examples = 0
    num_steps = 0

    # used for computing metrics
    all_golds = []
    all_preds = []

    eval_bar = tqdm(dataloader, total=len(dataloader), ascii=True)
    model.eval()
    for step, batch in enumerate(eval_bar):
        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = batch.get("labels")

        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=args.max_target_length,
                num_beams=args.num_beams,
            )

            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            all_preds.extend(preds)
            all_golds.extend(labels)

        num_examples += len(labels)
        num_steps += 1

    results = compute_acc_for_text(preds=all_preds, golds=all_golds, prefix=split)
    results.update({
        f"{split}/num_examples": num_examples,
        f"{split}/num_steps": num_steps,
    })
    logging.info(results)

    if split == 'test':
        logging.info(f"Saving {split} results and details...")
        # save results to json file
        save_dir = os.path.join(
            args.eval_dir, f"valid_epoch_{epoch}" if split == "val" else "test"
        )
        os.makedirs(save_dir)
        f1_reports = f1_scores(all_golds, all_preds)
        with open(os.path.join(save_dir, "results.json"), mode="w", encoding="utf-8") as f:
            json.dump(f1_reports, f)
            json.dump(results, f)

        # save summary
        inputs = [example.text for example in examples]
        input_pairs = [example.text_pair for example in examples]
        assert len(inputs) == len(input_pairs) == len(all_preds) == len(all_golds)
        print('-----------------', len(inputs), len(all_preds), '-------------------------------')
        df = pd.DataFrame(
            list(zip(inputs, input_pairs, all_golds, all_preds)),
            columns=["Input", "Input Pair", "Gold", "Pred"],
        )
        save_eval_details(save_dir=save_dir, detail_df=df)
        logging.info(f"{split.capitalize()} results and details are saved to {save_dir}")

    model.train()

    return results


def add_args(parser):
    parser.add_argument("--model_name",
                        type=str,
                        default="facebook/bart-base",
                        help="Model identifier.")
    parser.add_argument("--dataset_root",
                        type=str,
                        default=path,
                        help="Root of the dataset.")
    parser.add_argument("--run_name",
                        type=str,
                        default=None,
                        help="The name of run.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch size.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=20,
                        help="Total number of training epochs.")
    parser.add_argument("--max_train_steps",
                        type=int,
                        default=None,
                        help="Maximum training steps.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=5e-5,
                        help="Learning rate.")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0,
                        help="Weight decay rate.")
    parser.add_argument("--warmup_steps",
                        type=float,
                        default=0.1,
                        help="Warmup steps.")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Gradient accumulation steps.")

    parser.add_argument("--max_input_length",
                        type=int,
                        default=128,
                        help="Max length of input sequence.")
    parser.add_argument("--max_target_length",
                        type=int,
                        default=16,
                        help="Max length of output sequence.")
    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=1.0,
                        help="Max gradient norm.")
    parser.add_argument("--patience",
                        type=int,
                        default=5,
                        help="Early stopping patience.")
    parser.add_argument("--num_beams",
                        type=int,
                        default=5,
                        help="Beam search width.")

    parser.add_argument("--random_seed",
                        type=int,
                        default=42,
                        help="Random seed.")
    parser.add_argument("--fp16",
                        type=bool,
                        default=True,
                        help="Use mixed precision.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action="store_true",
                        help="Disable cuda and use cpu.")
    parser.add_argument("--cuda_visible_devices",
                        type=str,
                        default=None,
                        help="Visible cuda device ids, None to use all.")
    parser.add_argument("--logging_steps",
                        type=int,
                        default=100,
                        help="Step intervals to log training state.")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register("type", "bool", lambda v: v.lower() in ["yes", "true", "t", "1", "y"])

    add_args(parser)
    args = parser.parse_args()

    # prepare some preliminary arguments
    if args.run_name is None:
        args.run_name = "default"
    args.run_name = "{}_{}".format(args.run_name, time.strftime("%Y%m%d_%H%M%S", time.localtime()))

    # outputs and savings
    args.output_dir = os.path.join("..", "bart_outputs", args.run_name)  # root of outputs/savings
    args.model_dir = os.path.join(args.output_dir, "models")  # dir of saving models
    args.eval_dir = os.path.join(args.output_dir, "evaluations")  # dir of saving evaluation results
    for d in [args.model_dir, args.model_dir, args.eval_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # logging, log to both console and file, log debug-level to file
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    # terminal printing
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level=logging.INFO)
    logger.addHandler(console)
    # logging file
    file = logging.FileHandler(os.path.join(args.output_dir, "logging.log"))
    file.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s")
    file.setFormatter(formatter)
    logger.addHandler(file)

    logger.info("*" * 10 + " INITIALIZING " + "*" * 10)

    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Use CUDA: {args.use_cuda}")
    if args.use_cuda:
        if args.cuda_visible_devices is not None:
            logging.info(f"Visible CUDA device ids: {args.cuda_visible_devices}")
        args.num_devices = torch.cuda.device_count()
        logging.info(f"Number of available gpus: {args.num_devices}")
    else:
        args.num_devices = 1

    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    logging.info(f"Device: {args.device}")

    args.fp16 = args.fp16 and args.use_cuda
    logging.info(f"Use fp16 mixed precision: {args.fp16}")

    # set random seed
    if args.random_seed > 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    # log command and configs
    logger.debug("COMMAND: {}".format(" ".join(sys.argv)))

    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(args).items():
        config_table.add_row([config, str(value)])
    logger.debug("Configurations:\n{}".format(config_table))

    # --------------------------------------------------
    # Prepare tokenizer and model
    # --------------------------------------------------
    logging.info(f"***** Loading tokenizer and model *****")
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    model_summary(model)
    model.to(args.device)
    if args.num_devices > 1:
        model = torch.nn.DataParallel(model)

    # --------------------------------------------------
    # Train and valid
    # --------------------------------------------------
    model = train(args, model=model, tokenizer=tokenizer)
    torch.cuda.empty_cache()

    # --------------------------------------------------
    # test
    # --------------------------------------------------
    test_results = test(args, model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
