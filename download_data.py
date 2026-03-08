from sklearn.datasets import fetch_20newsgroups
from pathlib import Path

print("Fetching 20 Newsgroups dataset...")
dataset = fetch_20newsgroups(subset='all', remove=())

root = Path("20_newsgroups")
root.mkdir(exist_ok=True)

for i, (text, target) in enumerate(zip(dataset.data, dataset.target)):
    cat_name = dataset.target_names[target]
    cat_dir = root / cat_name
    cat_dir.mkdir(exist_ok=True)

    dest = cat_dir / str(i)
    with open(dest, 'w', encoding='latin-1', errors='replace') as f:
        f.write(text)

print(f"Done! {len(dataset.data)} posts written to {root.resolve()}")