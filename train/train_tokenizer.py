import json
from collections import Counter
from pathlib import Path

import numpy as np
from paths import *
from tokenizers import ByteLevelBPETokenizer

files = []
langs = []
counts = Counter()

with open(SPLIT_FILE) as fp:
    splits = json.load(fp)

for file, meta in splits["train_gru"].items():
    file = Path(file)

    if not file.exists():
        continue

    origin = meta["origin"]
    if origin == "tg":
        files.append(file)
        counts["tg"] += 1
    else:
        lang = meta["lang"]
        if lang not in counts or counts[lang] < 500 or Path(file).suffix == ".txt":
            files.append(file)
            counts[lang] += 1

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[file.as_posix() for file in files],
    vocab_size=2**15,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<s>",
        "<unk>",
    ],
    show_progress=True,
)

tokenizer.save_model(ARTIFACTS.as_posix(), "tokenizer")

tokens = [tokenizer.decode([i]) for i in range(tokenizer.get_vocab_size())]

num_tokens = len(tokens)
offsets = [0] + [len(token.encode("utf8")) + 1 for token in tokens]
offsets = np.cumsum(offsets).tolist()
total_len = offsets[-1]
offsets = offsets[:-1]

with open(RESOURCES / "tokenizer_vocab.bin", "wb") as file:
    file.write(num_tokens.to_bytes(length=4, byteorder="little"))
    file.write(total_len.to_bytes(length=4, byteorder="little"))

    for offset in offsets:
        file.write(offset.to_bytes(length=8, byteorder="little"))

    for token in tokens:
        file.write(token.encode("utf8"))
        file.write(b"\0")