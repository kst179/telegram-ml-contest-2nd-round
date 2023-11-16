from pathlib import Path

import torch
import tqdm
from gh_dataset import GHDataset
from tensorboardX import SummaryWriter
from tokenizers import ByteLevelBPETokenizer
from torch import nn
from torch.utils.data.dataloader import DataLoader

from .gh_dataset import MixedDataset
from .gru_model import NetworkConfig, Network

tokenizer = ByteLevelBPETokenizer(
    "../artifacts/tokenizer-vocab.json", "../artifacts/tokenizer-merges.txt"
)

output_dir = Path("../artifacts/gru_weights")
output_dir.mkdir(exist_ok=True, parents=True)

pad_token = tokenizer.token_to_id("<pad>")
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
    ids, labels = zip(*batch)

    max_len = max(len(i) for i in ids)
    max_len = min(max_len, max_seq_len)
    max_len = (max_len + 15) // 16 * 16

    ids = torch.tensor(
        [i[:max_len] + [pad_token for j in range(max_len - len(i))] for i in ids]
    )
    mask = ids != pad_token
    last_element = mask.sum(dim=1) - 1
    labels = torch.tensor([label.value - 1 for label in labels])

    return Batch(
        {
            "ids": ids,
            "mask": mask,
            "last_element": last_element,
            "labels": labels,
        }
    )


def train():
    config = NetworkConfig(
        num_classes=2,
        vocab_size=2**15,
        hidden_dim=112,
        num_layers=3,
        bidirectional=True,
        num_threads_per_dir=1,
    )

    train_dataset = MixedDataset(split="train_gru", tokenize=True, subsample_lines=True)
    val_dataset = MixedDataset(split="test", tokenize=True, subsample_lines=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn
    )

    model = Network()
    model.cuda()

    writer = SummaryWriter()

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    cross_entropy = nn.CrossEntropyLoss()
    grad_accum_steps = 2

    step = 0

    for epoch in tqdm.trange(200):
        model.train()
        cross_entropy.reduction = "mean"
        for batch in tqdm.tqdm(train_dataloader):
            batch.to("cuda")

            logits = model(batch["ids"], batch["last_element"])
            loss = cross_entropy(logits, batch["labels"])

            loss.backward()
            if step % grad_accum_steps == 0:
                optim.step()
                optim.zero_grad()

            writer.add_scalar("train_loss", loss.item(), step)
            step += 1

        avg_loss = 0
        avg_acc = 0
        n_batches = 0

        model.eval()
        cross_entropy.reduction = "sum"
        for batch, _ in zip(val_dataloader, tqdm.trange(1000)):
            batch.to("cuda")

            with torch.no_grad():
                logits = model(batch["ids"], batch["last_element"])

            loss = cross_entropy(logits, batch["labels"])
            avg_loss += loss.item()
            avg_acc += (logits.argmax(dim=-1) == batch["labels"]).float().sum().item()
            n_batches += batch["ids"].shape[0]

        avg_loss /= n_batches
        avg_acc /= n_batches

        writer.add_scalar("val_loss", loss, step)
        writer.add_scalar("val_acc", avg_acc, step)

        torch.save(model.state_dict(), output_dir / f"model_{epoch}.pth")


if __name__ == "__main__":
    train()