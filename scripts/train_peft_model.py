from dotenv import load_dotenv

load_dotenv()

import argparse
import gc
import json
import math
import os

import torch

from src.datasets import datasets
from src.models import models
from src.optimizers import AdamW
from src.trainers import SupervisedTrainer
from src.utils import save_config, set_seed, seed_worker, parse_args_json


def _build_parser():
    # ----- Parser -----
    parser = argparse.ArgumentParser(description="PEFT trainer and evaluator.")

    # ----- Dataset -----
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="cifar100",
        help="Dataset name.",
    )

    # ----- Backbone -----
    parser.add_argument(
        "--backbone-name",
        type=str,
        default="vit_b16",
        choices=["vit_b16"],
        help="Which backbone to use.",
    )

    # ----- PEFT method -----
    parser.add_argument(
        "--peft-name",
        type=str,
        default="vpt",
        choices=["vpt", "lora", "adaptformer", "ssf"],
        help="Which PEFT method to use.",
    )
    parser.add_argument(
        "--peft-config",
        type=json.loads,
        default={},
        help="Config for the PEFT method.",
    )

    # ----- Training -----
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Effective batch size after gradient accumulation.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=8,
        help="Mini-batch size processed per GPU step.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=25,
        help="Minimum number of epochs before early stopping.",
    )
    parser.add_argument(
        "--start-lr",
        type=float,
        default=1e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--ref-lr",
        type=float,
        default=1e-3,
        help="Reference learning rate.",
    )
    parser.add_argument(
        "--final-lr",
        type=float,
        default=1e-8,
        help="Final learning rate.",
    )
    parser.add_argument(
        "--ref-wd",
        type=float,
        default=1e-3,
        help="Reference weight decay.",
    )
    parser.add_argument(
        "--final-wd",
        type=float,
        default=1e-3,
        help="Final weight decay.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Number of warmup epochs.",
    )
    parser.add_argument(
        "--save-ckpt",
        action="store_true",
        help="Whether to save model weights during training.",
    )
    parser.add_argument(
        "--save-ckpt-interval",
        type=int,
        default=0,
        help="Intermediate checkpoint interval. 0 disables it.",
    )
    parser.add_argument(
        "--run-seeds",
        type=json.loads,
        default=[0],
        help="List of seeds to run. Example: '[0, 1, 2]'.",
    )
    parser.add_argument(
        "--eval-on-test",
        action="store_true",
        help="Also evaluate on the test set after validation.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Estimate bootstrap accuracy confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-num-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--bootstrap-ci",
        type=float,
        default=95.0,
        help="Bootstrap confidence interval percentage.",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device.",
    )

    # ----- IO -----
    parser.add_argument(
        "--save-dir",
        type=str,
        default="experiments/",
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Experiment subdirectory name.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Run experiment even if evaluation results already exist.",
    )
    return parser


def _parse_args():
    parser = _build_parser()
    return parse_args_json(parser)


def _validate_args(args):
    if args.batch_size <= 0 or args.gpu_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")

    if args.batch_size % args.gpu_batch_size != 0:
        raise ValueError("batch_size must be divisible by gpu_batch_size.")

    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")

    if (
        not isinstance(args.run_seeds, list)
        or not args.run_seeds
        or any(
            not isinstance(seed, int) or isinstance(seed, bool)
            for seed in args.run_seeds
        )
    ):
        raise ValueError("run_seeds must be a non-empty list of integers.")


def _train_seed(args, run_seed: int):
    # ----- Save dir and name -----
    save_dir = args.save_dir
    save_name = (
        f"{args.save_name}_seed-{run_seed}"
        if args.save_name is not None
        else f"seed-{run_seed}"
    )
    run_dir = os.path.join(save_dir, save_name)
    val_eval_path = os.path.join(run_dir, "train_summary.json")
    test_eval_path = os.path.join(run_dir, "test_acc_eval.json")

    if (
        not args.override
        and os.path.isfile(val_eval_path)
        and (not args.eval_on_test or os.path.isfile(test_eval_path))
    ):
        print(f"Experiment already done: {run_dir}. Skipping.")
        return

    # ----- Setup -----
    accum_steps = args.batch_size // args.gpu_batch_size
    set_seed(run_seed)

    # ----- Dataset metadata -----
    dataset_cls = datasets[args.dataset_name]
    img_size = dataset_cls.IMG_SIZE
    n_classes = dataset_cls.N_CLASSES

    # ----- Dataloaders -----
    dataset = dataset_cls()

    train_loader = dataset.get_dataloader(
        split="train",
        batch_size=args.gpu_batch_size,
        worker_init_fn=seed_worker,
    )
    val_loader = dataset.get_dataloader(
        split="val",
        batch_size=args.gpu_batch_size,
        worker_init_fn=seed_worker,
    )

    test_loader = None
    if args.eval_on_test:
        test_loader = dataset.get_dataloader(
            split="test",
            batch_size=args.gpu_batch_size,
            worker_init_fn=seed_worker,
        )

    # ----- Backbone -----
    backbone = models[args.backbone_name]["backbone"](img_size=img_size)
    backbone.freeze()
    backbone.eval()

    # ----- PEFT model -----
    model = models[args.backbone_name][args.peft_name](
        backbone=backbone,
        n_classes=n_classes,
        **args.peft_config,
    )
    model = model.to(args.device)

    # ----- Optimizer and schedulers -----
    epoch_len = math.ceil(len(train_loader) / accum_steps)
    adamw = AdamW(
        models=model,
        use_lr_scheduler=(args.start_lr != args.ref_lr or args.ref_lr != args.final_lr),
        use_wd_scheduler=args.ref_wd != args.final_wd,
        warmup_steps=args.warmup_epochs * epoch_len,
        start_lr=args.start_lr,
        ref_lr=args.ref_lr,
        final_lr=args.final_lr,
        ref_wd=args.ref_wd,
        final_wd=args.final_wd,
        T_max=args.epochs * epoch_len,
    )
    optimizer = adamw.get_optimizer()
    lr_scheduler = adamw.get_lr_scheduler()
    wd_scheduler = adamw.get_wd_scheduler()

    # ----- Trainer -----
    trainer = SupervisedTrainer(
        model=model,
        save_dir=run_dir,
    )

    # ----- Save config -----
    save_config(
        config={
            **vars(args),
            "run_seed": run_seed,
            "dataset": vars(dataset),
            "model": vars(model),
            "optimizer": vars(optimizer),
            "trainer": vars(trainer),
            "param_count": model.get_param_count(),
        },
        path=os.path.join(run_dir, "config.json"),
    )

    # ----- Train -----
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        accum_steps=accum_steps,
        early_stopping=args.early_stopping,
        patience=args.patience,
        device=args.device,
        save_ckpt=args.save_ckpt,
        save_ckpt_interval=args.save_ckpt_interval,
    )

    # ----- Optionally evaluate on test -----
    if args.eval_on_test:
        trainer.evaluate_acc(
            loader=test_loader,
            topk=(1,),
            save_name="test_acc_eval.json",
            bootstrap=args.bootstrap,
            bootstrap_num_samples=args.bootstrap_num_samples,
            bootstrap_ci=args.bootstrap_ci,
            bootstrap_seed=run_seed,
            device=args.device,
        )

    # ----- Cleanup -----
    del train_loader, val_loader, test_loader
    del model, backbone
    del adamw, optimizer, lr_scheduler, wd_scheduler
    del trainer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = _parse_args()
    _validate_args(args)

    for run_seed in args.run_seeds:
        print(f"\n===== Running seed {run_seed} =====")
        _train_seed(args, run_seed)


if __name__ == "__main__":
    main()
