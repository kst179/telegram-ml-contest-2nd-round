import ctypes

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from train.gh_dataset import MixedDataset
from train.languages_list import Languages
from train.paths import *


class CTGLang:
    def __init__(self, path = None):
        path = path or BUILD / "libtglang.so"
        self.lib = ctypes.CDLL(path)
        self.lib.tglang_detect_programming_language.argtypes = [ctypes.c_char_p]
        self.lib.tglang_detect_programming_language.restype = ctypes.c_int

    def __call__(self, string):
        return self.lib.tglang_detect_programming_language(string.encode())


if __name__ == "__main__":
    model = CTGLang()

    test_dataset = MixedDataset(split="test", subsample_lines=True, tokenize=True)

    labels = []
    predictions = []

    pbar = tqdm(test_dataset)

    for i, (text, label) in enumerate(pbar):
        labels.append(label.value - 1)
        predictions.append(model(text))

        if i % 500 == 0:
            acc = accuracy_score(labels, predictions)
            pbar.set_description(f"acc: {acc:.5}")

    labels = np.array(labels)
    predictions = np.array(predictions)

    np.save("../artifacts/final_model_predictions.npy", np.stack((labels, predictions)))

    cm = confusion_matrix(labels, predictions, labels=list(range(len(Languages))))
    np.save("../artifacts/confusion_matrix.npy", cm)

    print((labels == predictions).mean())
