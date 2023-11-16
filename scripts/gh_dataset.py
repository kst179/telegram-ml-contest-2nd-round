import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import loglaplace

from .ctokenizer import CTokenizer
from .languages_list import BinLabels, Languages, string_to_enum
from .paths import *

# estimated parameters of log log distribution of the texts len in the tg data
ALPHA = 70.53031203267946
BETA = 1.2777589461265924

class MixedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        gh_dir=DATA / "gh_data",
        tg_dir=DATA / "tg_data",
        split_file=SPLIT_FILE,
        split=None,
        mode="binary",
        tokenize=False,
        subsample_lines=True,
        max_len = 8192,
    ):
        self.gh_dir = gh_dir
        self.tg_dir = tg_dir
        self.subsample_lines = subsample_lines
        self.max_len = max_len
        self.tokenize = tokenize
        self.split = split
        self.mode = mode

        if split is not None:
            with open(split_file, "r") as file:
                splits = json.load(file)

            self.files = []
            self.origins = []
            self.langs = []
            self.is_code = []

            for file, meta in splits[split].items():
                file = Path(file)

                if not file.exists() or (mode != "binary" and meta["lang"] is None):
                    continue

                self.files.append(file)
                self.origins.append(meta["origin"])
                self.langs.append(Languages.from_string(meta["lang"]).value
                                  if meta["lang"] is not None
                                  else None)
                self.is_code.append(BinLabels(int(meta["is_code"])).value)
        else:
            gh_files = sorted(gh_dir.glob("*/*"))
            tg_files = sorted(tg_dir.glob("*/*"))

            origins = ["gh" for _ in gh_files] + ["tg" for _ in tg_files]
            is_code = []
            langs = []

            for file in gh_files:
                lang = Languages.from_string(file.parent.name)
                langs.append(lang.value)
                is_code.append(int(lang != Languages.OTHER))

            for file in tg_files:
                code_or_other = file.stem.split("-")[-1]
                langs.append(Languages.OTHER if code_or_other == "OTHER" else None)
                is_code.append(int(code_or_other == "CODE"))
            
            self.files = gh_files + tg_files
            self.origins = origins
            self.is_code = is_code
            self.langs = langs

        if tokenize:
            self.tokenizer = CTokenizer()

    def subsample_text(self, text: str) -> str:
        uniform = np.random.rand()
        target_length = (1 / uniform - 1)**(-1 / BETA) * ALPHA

        lines = text.split("\n")

        num_trys_left = 5

        while True:
            start_line = np.random.choice(max(len(lines) - 5, 1))
            current_length = 0
            sampled_lines = []

            for line_idx in range(start_line, len(lines)):
                line = lines[line_idx]
                next_length = current_length + len(line)

                if next_length >= target_length and next_length - target_length > target_length - current_length:
                    break

                sampled_lines.append(line)
                current_length += len(line)

                if current_length >= target_length:
                    break

            sampled_text = "\n".join(sampled_lines)
            
            if len(sampled_text.strip()) != 0:
                return sampled_text[:self.max_len]
            
            if num_trys_left == 0:
                break

            num_trys_left -= 1
    
        return text[:self.max_len]


    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]
        origin = self.origins[item]

        if self.mode == "binary":
            label = self.is_code[item]
        else:
            label = self.langs[item]

        text = file.read_text()
        if self.subsample_lines and origin == "gh":
            text = self.subsample_text(text)

        if self.tokenize:
            text = self.tokenizer.encode(text)

        return text, label, origin

class GHDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir=Path("../data/files"),
        split_file=Path("../data/splits.json"),
        split=None,
        tokenize=False,
        subsample_lines=False,
        max_num_lines=1024,
    ):
        self.dir = dir
        self.subsample_lines = subsample_lines
        self.max_num_lines = max_num_lines
        self.split = split

        if split is not None:
            with open(split_file, "r") as file:
                splits = json.load(file)
            self.files = sorted(Path(file) for file in splits[split])
        else:
            self.files = sorted(dir.glob("*/*"))

        self.labels = [
            Languages.from_string(filepath.parent.name) for filepath in self.files
        ]

        self.tokenize = tokenize
        if tokenize:
            self.tokenizer = CTokenizer()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        content = self.files[item].read_text()
        label = self.labels[item]

        if self.subsample_lines:
            lines = content.split("\n")

            if len(lines) > 5:
                for _ in range(5):
                    size = np.random.randint(1, min(len(lines), self.max_num_lines))
                    first = np.random.randint(0, len(lines) - size)

                    content = "\n".join(lines[first : first + size]).strip()
                    if content:
                        break

            if not content:
                content = "\n".join(lines)

        if self.tokenize:
            content = self.tokenizer.encode(content)

        return content, label
