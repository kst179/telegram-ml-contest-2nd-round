from train.paths import *
from train.gru_model import Network, NetworkConfig
from train.ctokenizer import CTokenizer
from train.languages_list import Languages

import torch

if __name__ == "__main__":
    tokenizer = CTokenizer()

    config = NetworkConfig(
        num_classes=2,
        vocab_size=2**15,
        hidden_dim=104,
        num_layers=3,
        bidirectional=True,
        num_threads_per_dir=1,
    )

    model = torch.nn.DataParallel(Network.from_config(config))
    model.load_state_dict(torch.load("artifacts/gru_binary/model_21_finetune.pth"))
    model.module.save_binary(RESOURCES / "gru_binary.bin")

    config.num_classes = len(Languages)

    model = torch.nn.DataParallel(Network.from_config(config))
    model.load_state_dict(torch.load("artifacts/gru_lang/model_96.pth"))
    model.module.save_binary(RESOURCES / "gru_lang.bin")