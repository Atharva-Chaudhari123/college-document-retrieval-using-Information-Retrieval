import os
from collections import defaultdict
from src.preprocess import preprocess


def build_inverted_index(data_dir):
    index = defaultdict(list)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                tokens = preprocess(f.read())
                for token in set(tokens):
                    index[token].append(file)

    return dict(index)
