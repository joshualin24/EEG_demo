"""Train a DINO model from the EEGBCI dataset."""


import pytorch_lightning as pl
import os
from augs import Identity, TimeMasking
from models import DINOEncoder, DINOProjector, DINO
from eegbci import PretextDataModule


# Augmentations
COMMON_AUGMENTATIONS = (
    Identity(),
)
STUDENT_EXCLUSIVE_AUGMENTATIONS = (
    TimeMasking(40, 80),
)

# Model architecture
EMBEDDING_DIM = 32
HIDDEN_DIM = 1024
BOTTLENECK_DIM = 128
PROJECTION_DIM = 512

# Model framework
STUDENT_TEMPERATURE = 0.01
TEACHER_TEMPERATURE = 0.07
TEACHER_MOMENTUM = 0.998
CENTER_MOMENTUM = 0.9

# Training
PRETEXT_BATCH_SIZE = 64
PRETEXT_LEARNING_RATE = 5e-4
PRETEXT_EPOCHS = 10

LOGS_DIR = './logs'
os.makedirs(LOGS_DIR, exist_ok=True)


if __name__ == '__main__':
    # Self-supervised learning model
    student = DINOEncoder(EMBEDDING_DIM, HIDDEN_DIM, num_layers=6)
    teacher = DINOEncoder(EMBEDDING_DIM, HIDDEN_DIM, num_layers=6)
    student_head = DINOProjector(EMBEDDING_DIM, HIDDEN_DIM, BOTTLENECK_DIM, PROJECTION_DIM, num_layers=2)
    teacher_head = DINOProjector(EMBEDDING_DIM, HIDDEN_DIM, BOTTLENECK_DIM, PROJECTION_DIM, num_layers=2)
    model = DINO(
        student,
        teacher,
        student_head,
        teacher_head,
        COMMON_AUGMENTATIONS,
        STUDENT_EXCLUSIVE_AUGMENTATIONS,
        PROJECTION_DIM,
        STUDENT_TEMPERATURE,
        TEACHER_TEMPERATURE,
        TEACHER_MOMENTUM,
        CENTER_MOMENTUM,
        PRETEXT_LEARNING_RATE
    )

    # Data and trainer
    pretext_datamodule = PretextDataModule(PRETEXT_BATCH_SIZE)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='Validation/Loss', mode='min')
    logger = pl.loggers.TensorBoardLogger(save_dir=LOGS_DIR, name='pretext')
    trainer = pl.Trainer(
        callbacks=[checkpoint],
        logger=logger,
        gpus=1,
        max_epochs=PRETEXT_EPOCHS
    )

    # Training and evaluation
    trainer.fit(model, datamodule=pretext_datamodule)