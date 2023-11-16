from pathlib import Path

import numpy as np
import torch
import tqdm
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score
from tokenizers import ByteLevelBPETokenizer
from torch import nn
from torch.utils.data.dataloader import DataLoader

from .gh_dataset import MixedDataset
from .gru_model import Network, NetworkConfig
from .paths import ARTIFACTS

output_dir = ARTIFACTS / "gru_weights"
output_dir.mkdir(exist_ok=True, parents=True)

pad_token = 0
max_seq_len = 8192


class Batch(dict):
    def __init__(self, dict):
        self.update(dict)

    def to(self, device):
        items = []
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                items.append((k, v.to(device)))

        for k, v in items:
            self[k] = v

        return self


def collate_fn(batch):
    ids, labels, origins = zip(*batch)

    max_len = max(len(i) for i in ids)
    max_len = min(max_len, max_seq_len)
    max_len = (max_len + 15) // 16 * 16

    ids = torch.tensor(
        [i[:max_len] + [pad_token for j in range(max_len - len(i))] for i in ids]
    )
    mask = ids != pad_token
    last_element = mask.sum(dim=1) - 1
    labels = torch.tensor(labels)
    origins = list(origins)

    return Batch(
        {
            "ids": ids,
            "last_element": last_element,
            "labels": labels,
            "origins": origins,
        }
    )


class Trainer:
    def __init__(self, net_config: NetworkConfig, train_config: dict):
        self.net_cfg = net_config
        self.train_cfg = train_config
        self.train_cfg.update(vars(net_config))

        wandb.init(
            project="tg-contest",
            entity="kst179",
            config=self.train_cfg,
            name=self.train_cfg.get("run", None)
        )

        self.model = Network.from_config(self.net_cfg)
        self.model.cuda()

        train_dataset = MixedDataset(split="train_gru", tokenize=True, subsample_lines=True, mode=self.train_cfg["mode"])
        val_dataset = MixedDataset(split="test", tokenize=True, subsample_lines=True, mode=self.train_cfg["mode"])

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.train_cfg["batch_size"], shuffle=True, collate_fn=collate_fn
        )
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.train_cfg["batch_size"], shuffle=False, collate_fn=collate_fn
        )

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg["lr"])
        self.cross_entropy = nn.CrossEntropyLoss()

        self.step = 0
        self.epoch = 0

    def train_epoch(self):
        self.model.train()
        self.cross_entropy.reduction = "mean"

        for batch in tqdm.tqdm(self.train_dataloader):
            batch.to("cuda")

            logits = self.model(batch["ids"], last_elements=batch["last_element"])
            loss = self.cross_entropy(logits, batch["labels"])

            loss.backward()
            if self.step % self.train_cfg["grad_accum_steps"] == 0:
                self.optim.step()
                self.optim.zero_grad()

            wandb.log({"train_loss": loss.item()}, step=self.step)
            self.step += 1

    def validate_epoch(self):
        origins = []
        labels = []
        predictions = []
        scores = []

        self.model.eval()
        for batch, _ in zip(self.val_dataloader, tqdm.trange(1000)):
            batch.to("cuda")

            with torch.no_grad():
                logits = self.model(batch["ids"], batch["last_element"])

            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch["labels"].cpu().numpy().tolist())
            origins.extend(batch["origins"])

        labels = np.array(labels)
        predictions = np.array(predictions)
        scores = np.array(scores)
        origins = np.array(origins)
        tg_mask = origins == "tg"

        wandb.log({
            "epoch": self.epoch,
            "accuracy": accuracy_score(labels, predictions),
            "tg_accuracy": accuracy_score(labels[tg_mask], predictions[tg_mask]),
            "gh_accuracy": accuracy_score(labels[~tg_mask], predictions[~tg_mask]),

            "auc_roc": roc_auc_score(labels, scores[:, 1] 
                                             if self.train_cfg["mode"] == "binary" 
                                             else scores),
            "tg_auc_roc": roc_auc_score(labels[tg_mask], scores[tg_mask, 1] 
                                                         if self.train_cfg["mode"] == "binary" 
                                                         else scores[tg_mask]),
            "gh_auc_roc": roc_auc_score(labels[~tg_mask], scores[~tg_mask, 1]
                                                          if self.train_cfg["mode"] == "binary" 
                                                          else scores[~tg_mask]),
        }, step=self.step)

        wandb.log({
            "roc": wandb.plot.roc_curve(labels, scores),
            "tg_roc": wandb.plot.roc_curve(labels[tg_mask], scores[tg_mask]),
            "gh_roc": wandb.plot.roc_curve(labels[~tg_mask], scores[~tg_mask]),
        }, step=self.step)


    def train(self):
        for self.epoch in tqdm.trange(self.train_cfg["num_epochs"]):
            self.train_epoch()
            self.validate_epoch()

            model_path = output_dir / f"model_{self.epoch}.pth"
            torch.save(self.model.state_dict(), model_path)
            wandb.save(model_path.as_posix())
