from train.paths import *
from train.gru_trainer import Trainer
from train.gru_model import NetworkConfig

if __name__ == "__main__":
    net_config = NetworkConfig(
        num_classes=2,
        vocab_size=2**15,
        hidden_dim=104,
        num_layers=3,
        bidirectional=True,
        num_threads_per_dir=1,
    )

    train_cfg = {
        "num_epochs": 200,
        "lr": 1e-5,
        "mode": "binary",
        "batch_size": 8,
        "grad_accum_steps": 4,
        "run": "binary_finetune",
        "output_dir": "gru_bin_finetune2",
        "split": "extra",
        "test_split": "test_tg",
        "start_checkpoint": "./artifacts/gru_binary/model_99.pth",
    }

    trainer = Trainer(net_config, train_cfg)
    trainer.train()