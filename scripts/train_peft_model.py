from dotenv import load_dotenv

load_dotenv()

import argparse
import gc
import json
import torch
import os
import math

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
        default="vtab_caltech101",
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
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Don't train the backbone.",
    )
    parser.add_argument(
        "--load-model-path",
        type=str,
        default=None,
        help="Path to load model from.",
    )

    # ----- Training -----
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Effective total batch size after gradient accumulation (gpu_batch_size × accumulation steps).",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=8,
        help="Mini-batch size processed per GPU step (before gradient accumulation).",
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
        help="Early stopping patience (epochs).",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=25,
        help="Min number of training epochs to wait before early stopping.",
    )
    parser.add_argument(
        "--start-lr",
        type=float,
        default=1e-5,
        help="Start training learning rate.",
    )
    parser.add_argument(
        "--ref-lr",
        type=float,
        default=1e-3,
        help="Training learning rate.",
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
        help="Training weight decay.",
    )
    parser.add_argument(
        "--final-wd",
        type=float,
        default=1e-3,
        help="Final training weight decay.",
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
        help="Whether to save model weights.",
    )
    parser.add_argument(
        "--save-ckpt-interval",
        type=int,
        default=0,
        help="Interval for saving intermediate checlpoints. 0 disables it.",
    )
    parser.add_argument(
        "--run-seeds",
        type=json.loads,
        default=[0],
        help="List of seeds to run. Example: --seeds '[0, 1, 2]'",
    )
    parser.add_argument(
        "--eval-on-test",
        action="store_true",
        help="If set, also evaluate the model on the test set in addition to the validation set.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="If set, also estimate bootstrap mean accuracy and confidence intervals during evaluation.",
    )
    parser.add_argument(
        "--bootstrap-num-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples to use when estimating evaluation confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-ci",
        type=float,
        default=95.0,
        help="Bootstrap confidence interval level as a percentage (for example, 95.0 for a 95%% confidence interval).",
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
        help=f"Compute device (default auto: cuda).",
    )

    # ----- IO -----
    parser.add_argument(
        "--save-dir",
        type=str,
        default="experiments/",
        help="Where to save outputs.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Subdir where to save outputs.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Run experiment even if it has already been executed before.",
    )

    return parser


def _parse_args():
    parser = _build_parser()
    return parse_args_json(parser)


def _train_seed(args, run_seed: int):
    # ----- Save dir and name -----
    save_dir = args.save_dir
    save_name = (
        f"{args.save_name}_seed-{run_seed}"
        if args.save_name is not None
        else f"seed-{run_seed}"
    )

    if not args.override:
        eval_split = "test" if args.eval_on_test else "val"
        eval_path = os.path.join(save_dir, save_name, f"{eval_split}_acc_eval.json")

        if os.path.exists(eval_path):
            print(f"Experiment already done: found {eval_path}. Skipping.")
            return

    if args.batch_size < args.gpu_batch_size:
        raise ValueError("batch_size should be >= gpu_batch_size")

    if args.batch_size % args.gpu_batch_size != 0:
        raise ValueError("batch_size should be divisable by gpu_batch_size")

    accum_steps = args.batch_size // args.gpu_batch_size

    set_seed(run_seed)

    # ------- Dataloaders -------
    img_size = datasets[args.dataset_name].IMG_SIZE
    n_classes = datasets[args.dataset_name].N_CLASSES

    if args.eval_on_test:
        train_ds = datasets[args.dataset_name](split="train+val")
        train_loader = train_ds.get_dataloader(
            batch_size=args.gpu_batch_size,
            worker_init_fn=seed_worker,
        )

        val_ds = None
        val_loader = None

        test_ds = datasets[args.dataset_name](split="test")
        test_loader = test_ds.get_dataloader(
            batch_size=args.gpu_batch_size,
            worker_init_fn=seed_worker,
        )
    else:
        train_ds = datasets[args.dataset_name](split="train")
        train_loader = train_ds.get_dataloader(
            batch_size=args.gpu_batch_size,
            worker_init_fn=seed_worker,
        )

        val_ds = datasets[args.dataset_name](split="val")
        val_loader = val_ds.get_dataloader(
            batch_size=args.gpu_batch_size,
            worker_init_fn=seed_worker,
        )

        test_ds = None
        test_loader = None

    # ----- PEFT model -----
    model = models[args.backbone_name][args.peft_name](
        img_size=img_size,
        n_classes=n_classes,
        freeze_backbone=args.freeze_backbone,
        **args.peft_config,
    )

    if args.load_model_path:
        model.load_state_dict(
            torch.load(args.load_model_path, weights_only=True),
            strict=False,
        )

    # ------- Optimizer -------
    epoch_len = math.ceil(len(train_loader) / accum_steps)
    adamw = AdamW(
        models=model,
        use_lr_scheduler=args.start_lr != args.ref_lr or args.ref_lr != args.final_lr,
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

    # ------- Trainer -------
    trainer = SupervisedTrainer(
        model=model,
        save_dir=os.path.join(save_dir, save_name),
    )

    # ------- Save config -------
    save_config(
        config={
            **vars(args),
            **{
                "run_seed": run_seed,
                "train_ds": vars(train_ds),
                "val_ds": vars(val_ds) if not args.eval_on_test else None,
                "test_ds": vars(test_ds) if args.eval_on_test else None,
                "model": vars(model),
                "optimizer": vars(optimizer),
                "trainer": vars(trainer),
                "param_count": model.get_param_count(),
            },
        },
        path=os.path.join(save_dir, save_name, "config.json"),
    )

    # ------- Train -------
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

    # ------- Eval -------
    trainer.evaluate_acc(
        loader=test_loader if args.eval_on_test else val_loader,
        topk=(1,),
        save_name=f"{'test' if args.eval_on_test else 'val'}_acc_eval.json",
        bootstrap=args.bootstrap,
        bootstrap_num_samples=args.bootstrap_num_samples,
        bootstrap_ci=args.bootstrap_ci,
        bootstrap_seed=run_seed,
        device=args.device,
    )

    # ------- Cleanup -------
    del train_ds, val_ds, test_ds
    del train_loader, val_loader, test_loader
    del model
    del adamw, optimizer, lr_scheduler, wd_scheduler
    del trainer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = _parse_args()

    for run_seed in args.run_seeds:
        print(f"\n===== Training seed {run_seed} =====")
        _train_seed(
            args,
            run_seed,
        )


if __name__ == "__main__":
    main()
