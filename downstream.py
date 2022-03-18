"""Modules for downstream tasks."""


import torch
from torch import nn
import pytorch_lightning as pl
from typing import Optional


class LinearClassifier(pl.LightningModule):
    """Linear neural classifier with a pretrained representation encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        finetune: bool = False,
        num_classes: int = 2,
        learning_rate: float = 1e-4
    ):
        """
        Arguments
        ---------
        encoder: Pretrained representation encoder
        embedding_dim: Dimension of pretrained representation
        finetune: Whether to finetune the pretrained encoder during training
        num_classes: Number of classes to classify data into
        learning_rate: Learning rate during training
        """
        super().__init__()
        self.encoder = encoder
        if not finetune: encoder.requires_grad_(False)
        self.linear = nn.Linear(embedding_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
    
    def forward(self, data: torch.Tensor):
        return self.linear(self.encoder(data))
    
    def training_step(self, batch, batch_ind):
        loss, acc = self._shared_step(batch, batch_ind)
        self.log('Train/Loss', loss)
        self.log('Train/Accuracy', acc)
        return loss
    
    def validation_step(self, batch, batch_ind):
        loss, acc = self._shared_step(batch, batch_ind)
        self.log('Validation/Loss', loss)
        self.log('Validation/Accuracy', acc)
    
    def test_step(self, batch, batch_ind):  # return logs by trainer.test()
        loss, acc = self._shared_step(batch, batch_ind)
        self.log('loss', loss)
        self.log('accuracy', acc)
    
    def _shared_step(self, batch, batch_ind):
        """Return loss and accuracy of the current batch."""
        data, labels = batch
        scores = self.linear(self.encoder(data))
        loss = self.criterion(scores, labels)
        acc = (scores.argmax(-1) == labels).mean()
        return loss, acc
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return opt


class KNNClassifier:
    """K-nearest neighbors classifier with pretrained representation encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 2,
        num_neighbors: int = 20,
        tau: float = 0.07,
        device: Optional[str] = None
    ):
        """
        Arguments
        ---------
        encoder: Pretrained representation encoder
        num_classes: Number of classes to classify data into
        num_neighbors: Number of nearest neighbors to infer labels of data
        tau: Sharpness parameter of scores for different classes
        device: Device to place encoder and data 
        """
        self.encoder = encoder.to(device=device).eval()
        self.num_classes = num_classes
        self.num_neighbors = num_neighbors
        self.tau = tau
        self.device = device
        # Memory banks for data reprsentation and labels
        self.bank = torch.empty(0, device=device)
        self.bank_labels = torch.empty(0).int()
    
    @torch.no_grad()
    def fit(self, batch: tuple):
        """Append memory bank by representation and labels of training data."""
        data, labels = batch
        data = data.to(device=self.device)
        self.bank = torch.cat([self.bank, self.encoder(data)], dim=0)
        self.bank_labels = torch.cat([self.bank_labels, labels], dim=0)
        return self
    
    @torch.no_grad()
    def predict(self, batch: tuple):
        """
        Return scores of different classes based on cosine similarity with
        training data representation.
        """
        data, _ = batch
        data = data.to(device=self.device)
        similarity = nn.functional.cosine_similarity(
            self.encoder(data).unsqueeze(1),
            self.bank.unsqueeze(0),
            dim=-1
        )
        weights, inds = torch.topk(similarity, self.num_neighbors, dim=1)
        scores = torch.zeros(  # with dimension (batch, classes, neighbors)
            data.size(0),
            self.num_classes,
            self.num_neighbors,
            device=data.device
        )
        mask_b = torch.arange(scores.size(0)).unsqueeze(1)
        mask_c = self.bank_labels[inds]
        mask_n = torch.arange(self.num_neighbors)
        scores[mask_b, mask_c, mask_n] = torch.exp(weights / self.tau)
        return scores.sum(-1)
    
    def accuracy(self, batch: tuple):
        """Return accuracy of the current batch."""
        scores = self.predict(batch)
        _, labels = batch
        acc = (scores.argmax(-1).cpu() == labels).float().mean().item()
        return acc


def main():
    """Empty main."""
    return


if __name__ == '__main__':
    main()