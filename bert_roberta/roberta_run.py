import argparse
import json
import os
import sys
import logging
import math
from tqdm import tqdm
import numpy as np
import time
import random
from prettytable import PrettyTable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers import get_scheduler, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from accelerate import Accelerator
from WarmupLR import WarmupLR

from model import ClassificationHead
from dataset import build_dataset, read_data
from utils import EarlyStopController, acc_for_multilabel, pretty_results, f1_scores, get_mypreds


logger = logging.getLogger()
folder = '../data_en'


def run_eval(
        args,
        model,
        dataloader,
        accelerator: Accelerator,
        examples,
        loss_fct,
        epoch,
        mode
) -> dict:
    eval_bar = tqdm(dataloader, total=len(dataloader), desc="Testing")

    # general statistics
    num_examples = 0
    num_steps = 0
    loss_list = []

    results = {}

    model.eval()
    preds = []
    golds = []
    my_preds = []
    for batch in eval_bar:
        with torch.no_grad():
            labels = batch.pop("labels")
            outputs = model(**batch, return_dict=True)

            logits = outputs["logits"]
            logits = torch.sigmoid(logits)
            my_preds = get_mypreds(my_preds, logits, args.threshold)
            loss = loss_fct(logits, labels)
            if args.num_gpus > 1:
                loss = loss.mean()
            loss_list.append(loss.item())

            pred = [np.where(logit > args.threshold)[0].tolist() for logit in logits.cpu().numpy()]
            preds.extend(pred)

            labels = accelerator.gather(labels)
            golds.extend(labels.squeeze().cpu().tolist())

            num_examples += batch["input_ids"].size(0)
            num_steps += 1

    assert len(preds) == len(golds) == len(examples)

    # compute acc
    results.update({
        "acc": acc_for_multilabel(preds=my_preds, golds=golds)
    })

    # save summary
    summary = []
    for pred, gold, example in zip(my_preds, golds, examples):
        summary.append({
            # "sentence": example.text,
            # "support": example.text_pair,
            "label_txt": example.label,
            "label_idx": gold,
            "prediction": pred
        })

    # other statistics
    if len(loss_list) > 0:
        results.update({f"test_loss": np.mean(loss_list)})
    results.update({
        f"test_num_examples": num_examples,
        f"test_num_steps": num_steps
    })
    if mode == 'test':
        # save predictions and golds
        save_dir = os.path.join(args.eval_dir, f"test_epoch_{epoch}")
        os.makedirs(save_dir)
        with open(os.path.join(save_dir, "predictions.txt"), mode="w", encoding="utf-8") as pred_f, \
                open(os.path.join(save_dir, "golds.txt"), mode="w", encoding="utf-8") as gold_f:
            for pred, gold in zip(my_preds, golds):
                pred_f.write(f"{str(pred)}\n")
                gold_f.write(f"{str(gold)}\n")
        with open(os.path.join(save_dir, "summary.json"), mode="w", encoding="utf-8") as f:
            json.dump(summary, f)
        # save results
        with open(os.path.join(save_dir, "results.json"), mode="w", encoding="utf-8") as f:
            json.dump(f1_scores(golds, my_preds), f)
            json.dump(results, f)

    model.train()

    return results


def run(args, accelerator: Accelerator):
    logger.info("*" * 10 + " LOADING " + "*" * 10)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = args.num_labels
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)  # , config=config
    model.classifier = ClassificationHead(
        hidden_size=config.hidden_size,
        num_labels=args.num_labels
    )

    # prepare data for training and validation
    train_examples, val_examples, test_examples = read_data(folder)
    train_dataset = build_dataset(
        examples=train_examples,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_dataset = build_dataset(
        examples=val_examples,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )

    test_dataset = build_dataset(
        examples=test_examples,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )

    logger.info(f"Data is loaded and prepared")

    logger.info("*" * 10 + " TRAINING " + "*" * 10)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, test_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    else:
        args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Scheduler and math around the number of training steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    total_batch_size = args.batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    completed_steps = 0
    early_stop = EarlyStopController(patience=args.patience, higher_is_better=True)
    loss_fct = nn.BCEWithLogitsLoss()

    test_acc, best_epoch = 0, 0
    for epoch in range(args.num_epochs):
        model.train()
        # warmup_scheduler.step(epoch)
        train_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"[epoch {epoch}, loss x.xxxx]")
        for step, batch in enumerate(train_bar):
            labels = batch.pop("labels")
            outputs = model(**batch, return_dict=True)

            logits = outputs["logits"]
            loss = loss_fct(logits, labels)

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                train_bar.set_description(f"[epoch {epoch}, loss {loss.item():.4f}]")

            if completed_steps >= args.max_train_steps:
                break

        logger.info("Start validation")
        val_results = run_eval(
            args,
            model=model,
            dataloader=val_dataloader,
            accelerator=accelerator,
            examples=val_examples,
            loss_fct=loss_fct,
            epoch=epoch,
            mode='val'
        )
        val_table = pretty_results(val_results)
        logger.info(f"End of validation at epoch {epoch}, results:\n{val_table}")
        early_stop(score=val_results["acc"], model=model, epoch=epoch)
        # save the model that achieves better acc than the previous epoch
        if early_stop.hit:
            save_best_dir = os.path.join(args.model_dir, f"better_acc")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(save_best_dir, save_function=accelerator.save)
            tokenizer.save_pretrained(save_best_dir)
            logger.info(f"The better acc model is saved to {save_best_dir}")
            test_results = run_eval(
                args,
                model=model,
                dataloader=test_dataloader,
                accelerator=accelerator,
                examples=test_examples,
                loss_fct=loss_fct,
                epoch=epoch,
                mode='test'
            )
            test_acc = test_results['acc']
            best_epoch = epoch
        if not early_stop.hit:
            logger.info(f"Early stopping counter: {early_stop.counter}/{early_stop.patience}")

        # last model
        save_last_dir = os.path.join(args.model_dir, "latest")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_last_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(save_last_dir)
        logger.info(f"The latest checkpoint is saved to {save_last_dir}")

        model.train()

        if early_stop.early_stop:
            logger.info(f"Early stopping is triggered")
            break

    last_results = run_eval(
        args,
        model=model,
        dataloader=test_dataloader,
        accelerator=accelerator,
        examples=test_examples,
        loss_fct=loss_fct,
        epoch=args.num_epochs,
        mode='test'
    )
    last_acc = last_results['acc']
    if last_acc > test_acc:
        test_acc = last_acc
        best_epoch = args.num_epochs

    torch.cuda.empty_cache()
    return test_acc, best_epoch


def main():
    # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register("type", "bool", lambda v: v.lower() in ["yes", "true", "t", "1", "y"])

    add_args(parser)
    args = parser.parse_args()

    # prepare some preliminary arguments
    if args.run_name is None:
        args.run_name = "default"
        args.short_run_name = "default"
    args.run_name = "{}_{}".format(args.run_name, time.strftime("%Y%m%d_%H%M%S", time.localtime()))

    # outputs and savings
    args.output_dir = os.path.join("..", "outputs", args.run_name)  # root of outputs/savings
    args.model_dir = os.path.join(args.output_dir, "models")  # dir of saving models
    args.eval_dir = os.path.join(args.output_dir, "evaluations")  # dir of saving evaluation results
    # args.run_dir = os.path.join(args.output_dir, "runs")  # dir of tracking running
    for d in [args.model_dir, args.model_dir, args.eval_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # logging, log to both console and file, log debug-level to file
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

    # set distribution and mixed precision, using `accelerate` package
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    args.use_cuda = torch.cuda.is_available()
    if args.use_cuda:
        accelerator = Accelerator(mixed_precision=args.mixed_precision)
    else:
        accelerator = Accelerator(cpu=True)
    args.device = accelerator.device
    args.num_gpus = accelerator.num_processes

    logger.info(accelerator.state)

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

    best_acc, best_epoch = run(args, accelerator)
    logger.info(f"best_acc: {best_acc}", f"best_epoch: {best_epoch}")


def add_args(parser):
    parser.add_argument("--model_name",
                        type=str,
                        default="roberta-base",
                        help="Model identifier.")
    parser.add_argument("--run_name",
                        type=str,
                        default=None,
                        help="Name of this run.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="Batch size.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=20,
                        help="Total number of training epochs.")
    parser.add_argument("--max_train_steps",
                        type=int,
                        default=None,
                        help="Maximum training steps.")
    parser.add_argument("--threshold",
                        type=float,
                        default=0.5,
                        help="Threshold to make predictions.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=5e-5,
                        help="Learning rate.")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0,
                        help="Weight decay rate.")
    parser.add_argument("--num_warmup_steps",
                        type=int,
                        default=0.001,
                        help="Warmup steps.")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Gradient accumulation steps.")

    parser.add_argument("--max_length",
                        type=int,
                        default=128,
                        help="Max length of input.")
    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=1.0,
                        help="Max gradient norm.")
    parser.add_argument("--patience",
                        type=int,
                        default=5,
                        help="Early stopping patience.")

    parser.add_argument("--num_labels",
                        type=int,
                        default=5,
                        help="Number of labels.")
    parser.add_argument("--random_seed",
                        type=int,
                        default=3407,
                        help="Random seed.")
    parser.add_argument("--mixed_precision",
                        type=str,
                        default="fp16",
                        help="Mixed precision type.")


if __name__ == "__main__":
    main()
