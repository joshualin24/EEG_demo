"""Modules of self-supervised learning modeling frameworks."""


import torch
from torch import nn
import pytorch_lightning as pl
import math
from typing import Optional, Type
from collections.abc import Sequence
from augs import RandomAugmentationPair


#################
###   DINO
#################


class DINOEncoder(nn.Module):
    """Representation encoder based on transformer."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_channels: int = 64,
        num_layers: int = 12,
        num_heads: int = 8,
        cls_token: Optional[torch.Tensor] = None,
        norm_layer: Type = nn.LayerNorm
    ):
        """
        Arguments
        ---------
        embedding_dim: Dimension of output embedding
        hidden_dim: Dimension of feedforward module
        num_channels: Number of channels in input data
        num_layers: Number of transformer encoder layers
        num_heads: Number of multi-attention heads
        cls_token: One-dimensional embedding of the [CLS] token
        norm_layer: Type of normalization layer applied to output embedding

        Note
        ----
        If specified, `cls_token` must have length of `embedding_dim`.
        """
        super().__init__()
        self.linear = nn.Linear(num_channels, embedding_dim, bias=False)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embedding_dim,
                num_heads,
                dim_feedforward=hidden_dim,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers
        )
        self.register_buffer(
            'token',
            cls_token if cls_token else torch.zeros(embedding_dim)
        )
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.norm = norm_layer(embedding_dim)
    
    def forward(self, data: torch.Tensor):
        """
        Output embedding with dimension `(batch, ... , embedding)` from input
        data with dimension `(batch, ..., channels, time steps)`.
        """
        embedding = self.prepare_tokens(data)
        embedding = embedding.flatten(end_dim=-3)  # flatten additional dim.
        embedding = self.transformer(embedding)
        embedding = embedding.unflatten(0, data.size()[:-2])  # restore dim.
        embedding = self.norm(embedding)
        return embedding[..., 0, :]  # embedding corresponding to [CLS] token
    
    def prepare_tokens(self, data: torch.Tensor):
        """
        Linearly combine data into embedding dimension, add [CLS] token, and
        perform positional encoding.
        """
        embedding = self.linear(data.transpose(-2, -1))
        cls_token = self.token[(None,) * (len(data.size())-1)]  # add dimension
        embedding = torch.cat([
            cls_token.expand(*data.size()[:-2], -1, -1),
            embedding
        ], dim=-2)
        embedding = self.pos_encoder(embedding)
        return embedding


class DINOProjector(nn.Module):
    """Projection head based on multilayer perceptron."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        projection_dim: int,
        num_layers: int = 3
    ):
        """
        Arguments
        ---------
        embedding_dim: Dimension of input embedding
        hidden_dim: Dimension of hidden layers in multilayer perceptron
        bottleneck_dim: Dimension of L-2 normlaization bottleneck
        projection_dim: Dimension of output projection
        num_layers: Number of layers of multilayer perceptron
        """
        super().__init__()
        if num_layers > 1:
            first = [nn.Linear(embedding_dim, hidden_dim), nn.GELU()]
            middle = [
                nn.Linear(hidden_dim, hidden_dim) if i % 2 == 0 else nn.GELU()
                for i in range(2 * (num_layers - 2))
            ]
            last = [nn.Linear(hidden_dim, bottleneck_dim), nn.GELU()]
            self.mlp = nn.Sequential(*first, *middle, *last)
        else:
            self.mlp = nn.Linear(embedding_dim, bottleneck_dim)
        self.linear = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, projection_dim, bias=False)
        )
    
    def forward(self, embedding: torch.Tensor):
        """Input and output have dimension `(batch, ..., embedding)`."""
        embedding = self.mlp(embedding)
        embedding = nn.functional.normalize(embedding, dim=-1)  # L2 norm.-ed
        embedding = self.linear(embedding)
        return embedding


class PositionalEncoding(nn.Module):
    """
    Position encoding layer for transformer-based modules.

    This implementation is adopted from the Pytorch tutorial
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(max_len).unsqueeze(1)
        factor = -math.log(10000.0) / d_model
        div_even = torch.exp(torch.arange(0, d_model, 2) * factor)
        div_odd = torch.exp((torch.arange(1, d_model, 2) - 1) * factor)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div_even)
        pe[0, :, 1::2] = torch.cos(pos * div_odd)
        self.register_buffer('pe', pe)

    def forward(self, data: torch.Tensor):
        """
        Input must have dimension `(batch, ..., sequence, embedding)`.
        """
        data = data + self.pe[:, :data.size(-2)]
        return self.dropout(data)


class StudentTeacherLoss(nn.Module):
    """Cross-entropy loss between student and teacher outputs after softmax."""

    def __init__(
        self,
        dim: int,
        num_std_augs: int,
        num_tch_augs: int,
        student_temperature: float,
        teacher_temperature: float,
        momentum: float = 0.9
    ):
        """
        Arguments
        ---------
        dim: Output dimension of student/teacher module
        num_std_augs: Number of augmentations passed to student module
        num_tch_augs: Number of augmentations passed to teacher module
        student_temperature: Sharpness parameter for student output
        teacher_temperature: Sharpness parameter for teacher output
        momentum: Update rate parameter of center buffer
        """
        super().__init__()
        self.register_buffer('center', torch.zeros(dim))
        self.register_buffer(
            'mask',
            (~torch.eye(num_tch_augs, num_std_augs, dtype=bool)).float()
        )
        self.std_tpr = student_temperature
        self.tch_tpr = teacher_temperature
        self.momentum = momentum

    def forward(
        self,
        std_out: torch.Tensor,
        tch_out: torch.Tensor,
        train: bool = True
    ):
        """
        Arguments
        ---------
        std_out: Output of student module
        tch_out: Output of teacher module
        train: Whether to update center buffer

        Note
        ----
        Output of student/teacher module must have dimension
        `(batch, augmentations, projection/embedding)`.
        """
        mask = self.mask  # to exclude same augmentation in student/teacher
        for _ in range(len(std_out.size())-2): mask = mask.unsqueeze(-1)
        tch_prob = nn.functional.softmax(
            (tch_out - self.center) / self.tch_tpr,
            dim=-1
        ).unsqueeze(2)
        std_log_prob = nn.functional.log_softmax(
            std_out / self.std_tpr,
            dim=-1
        ).unsqueeze(1)
        loss = (-mask * tch_prob * std_log_prob).sum() / tch_out.size(0)
        if train: self.update_center(tch_out)
        return loss
    
    @torch.no_grad()
    def update_center(self, tch_out: torch.Tensor):
        """Update center buffer used for teacher output."""
        self.center *= self.momentum
        self.center += (1 - self.momentum) * tch_out.flatten(end_dim=-2).mean(0)


class DINO(pl.LightningModule):
    """Self-supervised learning motivated by knowledge distillation."""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        student_head: nn.Module,
        teacher_head: nn.Module,
        common_augs: tuple,
        std_exclusive_augs: tuple,
        projection_dim: int,
        student_temperature: float,
        teacher_temperature: float,
        teacher_momentum: float,
        center_momentum: float,
        learning_rate: float,
        weight_decay: float = 0.01
    ):
        """
        Arguments
        ---------
        student: Encoder of student module
        teacher: Encoder of teacher module
        student_head: Projection head of student module
        teacher_head: Projection head of teacher module
        common_augs: Augmentations passed to both student and teacher modules
        std_exclusive_augs: Augmentations passed to student module only
        projection_dim: Output dimension of projection head
        student_temperature: Sharpness parameter for student output
        teacher_temperature: Sharpness parameter for teacher output
        teacher_momentum: Update rate parameter of teacher module
        center_momentum: Update rate parameter of dummy centers
        learning_rate: Learning rate of student module
        weight_decay: Weight decay when training student module

        Notes
        -----
        1. `teacher`/`teacher_head` must have the same architecture as
           `student`/`student_head`.
        2. `common_augs` and `std_exclusive_augs` should be mutually exclusive.
        3. `projection_dim` must match the dimension of output of
           `student_head`/`teacher_head`.
        """
        super().__init__()
        self.student = nn.Sequential(student, student_head)
        self.teacher = nn.Sequential(teacher, teacher_head)
        self.encoder = student
        self.std_augs = common_augs + std_exclusive_augs
        self.tch_augs = common_augs
        self.momentum = teacher_momentum
        self.criterion = StudentTeacherLoss(
            projection_dim,
            len(self.std_augs),
            len(self.tch_augs),
            student_temperature,
            teacher_temperature,
            center_momentum
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # Force teacher and student module to start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())
        # Disable gradient tracking for teacher module
        for p in self.teacher.parameters(): p.requires_grad_(False)
        # Enable manual backpropagation
        self.automatic_optimization = False

    def forward(self, data: torch.Tensor):
        """Return the learned representation of data."""
        return self.encoder(data)
    
    def training_step(self, batch, batch_ind):
        # Manually update student module
        opt = self.optimizers()
        opt.zero_grad()
        loss = self._shared_step(batch, batch_ind)
        self.manual_backward(loss)
        opt.step()
        # Update teacher module with exponential moving average
        for p_tch, p_std in zip(
            self.teacher.parameters(),
            self.student.parameters()
        ):
            p_tch.mul_(self.momentum).add_(p_std.detach(), alpha=1.0-self.momentum)
        # Log metric
        self.log('Train/Loss', loss)
        return loss
    
    def validation_step(self, batch, batch_ind):
        loss = self._shared_step(batch, batch_ind, train=False)
        self.log('Validation/Loss', loss)
    
    def _shared_step(self, batch, batch_ind, **kwargs):
        """Return loss of the current batch."""
        data, _ = batch
        augmented_std = torch.stack([aug(data) for aug in self.std_augs], dim=1)
        augmented_tch = torch.stack([aug(data) for aug in self.tch_augs], dim=1)
        loss = self.criterion(
            self.student(augmented_std),
            self.teacher(augmented_tch).detach(),
            **kwargs
        )
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(  # optimizer to train student module
            self.student.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return opt


#################
###   SeqCLR
#################


class ContrastiveLoss(nn.Module):
    """
    Constrative loss function in the SimCLR framework.

    Notes
    -----
    1. Full credit to https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/.
    2. See also Chen, Ting, et al. "A simple framework for contrastive learning
       of visual representations." International conference on machine learning.
       PMLR, 2020.
    """

    def __init__(self, batch_size: int, temperature: float = 0.5):
        """
        Arguments
        ---------
        batch_size: Batch size (as well as the size of candidate pool)
        temperature: Parameter of sharpness
        """
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer(
            "negatives_mask",
            (~torch.eye(batch_size*2, batch_size*2, dtype=bool)).float()
        )
            
    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor):
        """
        Return the Boltzmann distribution of positive pairs, where the energy
        function is the cosine similarity between embeddings.
        """
        z_i = torch.nn.functinal.normalize(emb_i, dim=1)
        z_j = torch.nn.functinal.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        # Compute cosine similarity between positive pairs
        similarity_matrix = torch.nn.functinal.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        # Compute loss value
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class SeqCLR(pl.LightningModule):
    """Contrastive learning on individual channels of EEG data."""

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        augmentations: Sequence,
        batch_size: int,
        temperature: float = 5e-2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        """
        Arguments
        ---------
        encoder: Encoder module that extract representation from data
        projector: Projection head module
        augmentations: Collection of augmentations of interests
        batch_size: Batch size of training/validation set
        temperature: Temperature of contrastive loss
        learning_rate: Learning rate of optimizer
        weight_decay: Weight decay of optimizer
        """
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.augmentation_pair = RandomAugmentationPair(augmentations)
        self.criterion = ContrastiveLoss(batch_size, temperature)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def forward(self, data: torch.Tensor):
        """
        Return the learned representation of data.

        Note
        ----
        Data must have dimensions `(batch, channels, ..., time steps)`.
        """
        repr = self.encoder(data.flatten(end_dim=1))
        return repr.unflatten(0, data.size()[:2])
    
    def training_step(self, batch, batch_ind):
        loss = self._shared_step(batch, batch_ind)
        self.log('Train/Loss', loss)
        return loss
    
    def validation_step(self, batch, batch_ind):
        loss = self._shared_step(batch, batch_ind)
        self.log('Validation/Loss', loss)
    
    def _shared_step(self, batch, batch_ind):
        """Return loss of the current batch."""
        data, = batch
        data = data.flatten(end_dim=1)  # flatten channels into the batch
        augmented_1, augmented_2 = self.augmentation_pair(data)
        loss = self.criterion(
            self.projector(self.encoder(augmented_1)),
            self.projector(self.encoder(augmented_2))
        )
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer