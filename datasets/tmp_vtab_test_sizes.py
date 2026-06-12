if __name__ == "__main__":
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from datasets import datasets

    vtab_names = sorted(name for name in datasets if name.startswith("vtab_"))

    for name in vtab_names:
        val_dataset = datasets[name](split="val")
        test_dataset = datasets[name](split="test")
        print(f"{name}: {len(val_dataset) + len(test_dataset)}")
