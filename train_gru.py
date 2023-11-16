from scripts.gru_trainer import Trainer
from scripts.gru_model import NetworkConfig

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
        "lr": 1e-4,
        "mode": "binary",
        "batch_size": 4,
        "grad_accum_steps": 2,
        "run": "test",
    }

    trainer = Trainer(net_config, train_cfg)
    trainer.train()