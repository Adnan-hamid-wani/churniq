"""Download and verify the IBM sample. Run from the project root."""
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.data_loader import DATA_PATH, download_bytes, load_dataset


def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = download_bytes()
    with NamedTemporaryFile(dir=DATA_PATH.parent, delete=False) as temp:
        temp.write(payload)
        path = Path(temp.name)
    try:
        data = load_dataset(path)
        if len(data) != 7043:
            raise ValueError("Expected 7,043 customer records.")
        path.replace(DATA_PATH)
    finally:
        path.unlink(missing_ok=True)
    print(f"Verified {len(data):,} records in {DATA_PATH.relative_to(DATA_PATH.parents[2])}")


if __name__ == "__main__":
    main()
