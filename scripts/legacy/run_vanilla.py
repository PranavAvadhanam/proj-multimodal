from pathlib import Path


if __name__ == "__main__":
    nb = Path(__file__).with_suffix(".ipynb")
    raise SystemExit(
        f"Runner migrated to notebook: {nb}\n"
        "Open and run cells in Jupyter/Colab. Legacy CLI copy lives in scripts/legacy/run_vanilla.py."
    )

