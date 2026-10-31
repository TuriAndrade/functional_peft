if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    import sys
    from pathlib import Path

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    import argparse
    import torch
    from datasets import datasets
    from collections import Counter

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-under-test",
        type=str,
        default=None,
        help="Dataset key to test. If omitted, tests all datasets.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print batch details while testing.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=3)

    args = parser.parse_args()

    if args.dataset_under_test is not None:
        assert args.dataset_under_test in datasets, (
            f"Unknown dataset: {args.dataset_under_test}. "
            f"Available datasets: {sorted(datasets.keys())}"
        )
        dataset_names = [args.dataset_under_test]
    else:
        dataset_names = sorted(datasets.keys())

    assert len(dataset_names) > 0, "No datasets found to test."

    for _ds in dataset_names:
        assert _ds in datasets, f"Dataset key not found: {_ds}"

        for split in ["train", "val", "train+val", "test"]:
            if args.verbose:
                print(f"\n=== {_ds} split={split} ===")

            dataset = datasets[_ds](split=split)

            assert hasattr(dataset, "IMG_SIZE"), f"{_ds} missing IMG_SIZE"
            assert hasattr(dataset, "N_CLASSES"), f"{_ds} missing N_CLASSES()"
            assert hasattr(dataset, "get_dataloader"), f"{_ds} missing get_dataloader()"
            assert len(dataset) > 0, f"{_ds}/{split} has zero examples"
            assert dataset.IMG_SIZE > 0, f"{_ds}/{split} invalid image size"
            assert dataset.N_CLASSES > 1, f"{_ds}/{split} invalid number of classes"

            if args.verbose:
                print(f"Split string: {dataset.split}")
                print(f"Expected num examples: {len(dataset)}")
                print(f"Image size: {dataset.IMG_SIZE}")
                print(f"Num classes: {dataset.N_CLASSES}")

            loader = dataset.get_dataloader(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=False,
                drop_last=False,
            )

            label_counter = Counter()
            num_seen = 0

            for batch_idx, (images, labels) in enumerate(loader):
                label_counter.update(labels.tolist())
                num_seen += labels.numel()

                if args.verbose:
                    print(f"\nBatch {batch_idx}")
                    print(f"images.shape: {tuple(images.shape)}")
                    print(f"labels.shape: {tuple(labels.shape)}")
                    print(f"labels: {labels.tolist()}")

                assert images.ndim == 4, (
                    f"{_ds}/{split}: expected images.ndim == 4, " f"got {images.ndim}"
                )
                assert images.shape[1:] == (
                    3,
                    dataset.IMG_SIZE,
                    dataset.IMG_SIZE,
                ), f"{_ds}/{split}: invalid image shape {tuple(images.shape)}"
                assert (
                    labels.ndim == 1
                ), f"{_ds}/{split}: expected labels.ndim == 1, got {labels.ndim}"
                assert (
                    labels.dtype == torch.long
                ), f"{_ds}/{split}: expected labels dtype torch.long, got {labels.dtype}"
                assert labels.numel() > 0, f"{_ds}/{split}: empty label batch"
                assert (
                    labels.min().item() >= 0
                ), f"{_ds}/{split}: negative label found: {labels.min().item()}"
                assert labels.max().item() < dataset.N_CLASSES, (
                    f"{_ds}/{split}: label {labels.max().item()} >= "
                    f"n_classes {dataset.N_CLASSES}"
                )

                if batch_idx + 1 >= args.max_batches:
                    break

            assert num_seen > 0, f"{_ds}/{split}: no samples yielded by DataLoader"
            assert len(label_counter) > 0, f"{_ds}/{split}: no labels counted"

            if args.verbose:
                print(f"\nSeen in checked batches: {num_seen}")
                print(f"Small label histogram: {dict(label_counter)}")

        print(f"[OK] {_ds}")

    print("\nDone.")
