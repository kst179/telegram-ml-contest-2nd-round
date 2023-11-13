from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

# model_path = "../artifacts/gru_weights/model_199.pth"

def pad_to_multiple_of_8(tensor, value=0):
    pad = (8 - tensor.shape[0]) % 8

    if len(tensor.shape) == 2:
        return torch.nn.functional.pad(tensor, (0, 0, 0, pad), value=value)
    if len(tensor.shape) == 1:
        return torch.nn.functional.pad(tensor, (0, pad), value=value)
    
    raise ValueError("Can pad only matrix or vector")

@dataclass
class NetworkConfig:
    num_classes: int = 2
    vocab_size: int = 2**15
    hidden_dim: int = 96
    num_layers: int = 2
    bidirectional: bool = True
    num_threads_per_dir: int = 1

class Network(nn.Module):
    @staticmethod
    def from_config(config: NetworkConfig) -> Network:
        return Network(
            config.num_classes,
            config.vocab_size,
            config.hidden_dim, 
            config.num_layers,
            config.bidirectional,
            config.num_threads_per_dir,
        )


    def __init__(self, num_classes=2, vocab_size=2**15, hidden_dim=96, num_layers=2, bidirectional=True, num_threads_per_dir=1):
        assert(hidden_dim % 8 == 0)

        super().__init__()
        num_directions = 2 if bidirectional else 1

        self.num_layers = num_layers
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_directions = num_directions
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_threads_per_dir = num_threads_per_dir

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
        self.grus = nn.ModuleList()
        
        for layer in range(num_layers):
            input_size = hidden_dim if layer == 0 else hidden_dim * num_directions 
            self.grus.append(
                nn.GRU(input_size=input_size, hidden_size=hidden_dim,
                       bidirectional=bidirectional, batch_first=True)
            )
        
        self.classifier = nn.Linear(hidden_dim * num_directions, num_classes)

    def forward(self, ids, last_elements=None, return_last_state=False):
        """ids: [batch_size, seq_len]"""

        if last_elements is None:
            last_elements = (ids != 0).sum(dim=1) - 1
        
        batch_size = ids.shape[0]

        # [batch_size, seq_len, hid_dim]
        features = self.embedding(ids)

        # [batch_size, seq_len, hid_dim * num_directions]
        for gru in self.grus:
            features, _ = gru(features)

        features = features.view(batch_size, -1, self.num_directions, self.hidden_dim)

        batch = range(batch_size)
        first_elements = [0 for _ in batch]
        fwd = [0 for _ in batch]
        bwd = [1 for _ in batch]

        if self.bidirectional:
            last_feature = torch.cat(
                (
                    features[batch, last_elements, fwd, :],    # last state in fwd dir
                    features[batch, first_elements, bwd, :],   # first state in bwd dir
                ), dim=1
            )
        else:            
            last_feature = features[batch, last_elements, fwd, :]

        # [batch_size, hid_dim] -> [batch_size, num_classes]
        logits = self.classifier(last_feature)

        if return_last_state:
            return logits, last_feature

        return logits
    
    def save_binary(self, filepath: Path):
        hidden_dim = self.hidden_dim
        dim_per_thread = hidden_dim // self.num_threads_per_dir
        order = []

        for thread_idx in range(self.num_threads_per_dir):
            start = dim_per_thread * thread_idx
            end = dim_per_thread * (thread_idx + 1)

            order.extend(range(start, end))
            order.extend(range(hidden_dim + start, hidden_dim + end))
            order.extend(range(hidden_dim * 2 + start, hidden_dim * 2 + end))

        with open(filepath, "wb") as file:
            write_int = lambda x: file.write(x.to_bytes(length=4, byteorder="little"))
            write_tensor = lambda x: file.write(x.detach().cpu().numpy().tobytes())

            write_int(self.num_layers)
            write_int(int(self.bidirectional))
            write_int(self.num_threads_per_dir)

            write_int(self.hidden_dim)
            write_int(self.vocab_size)
            write_int((self.num_classes + 7) // 8 * 8)

            state_dict = self.state_dict()

            embeddings = self.embedding.weight.data
            weights_i = self.grus[0].weight_ih_l0
            bias_i = self.grus[0].bias_ih_l0

            write_tensor((embeddings @ weights_i.T + bias_i)[:, order])

            if self.bidirectional:
                weights_i = self.grus[0].weight_ih_l0_reverse
                bias_i = self.grus[0].bias_ih_l0_reverse

                write_tensor((embeddings @ weights_i.T + bias_i)[:, order])                

            for layer in range(self.num_layers):
                for dir in range(self.num_directions):
                    attr = f"{{}}_hh_l0{'_reverse' if dir else ''}"

                    weights_h = getattr(self.grus[layer], attr.format("weight"))
                    bias_h = getattr(self.grus[layer], attr.format("bias"))

                    write_tensor(weights_h[order])
                    write_tensor(bias_h[order])

                    if layer != 0:
                        attr = f"{{}}_ih_l0{'_reverse' if dir else ''}"

                        weights_i = getattr(self.grus[layer], attr.format("weight"))
                        bias_i = getattr(self.grus[layer], attr.format("bias"))
                        
                        write_tensor(weights_i[order])
                        write_tensor(bias_i[order])
            
            classifier_weight = state_dict["classifier.weight"]
            classifier_bias = state_dict["classifier.bias"]

            classifier_weight = pad_to_multiple_of_8(classifier_weight, 0)
            classifier_bias = pad_to_multiple_of_8(classifier_bias, -torch.inf)

            print(classifier_weight.shape)
            print(classifier_bias.shape)

            write_tensor(classifier_weight)
            write_tensor(classifier_bias)


# state_dict = torch.load(model_path, map_location="cpu")
# model = Network(num_layers=4, hidden_dim=96)
# model.load_state_dict(torch.load("test.pth"))
# state_dict = model.state_dict()
# num_threads_per_dir = 1

# model.load_state_dict(state_dict)