"""Evaluate a pretrained DINO model with the EEGBCI data."""


import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eegbci import DownstreamDataModule, EVENT_ID
from downstream import KNNClassifier


DOWNSTREAM_BATCH_SIZE = 64
DOWNSTREAM_NUM_STEPS = 640
TASKS = ['annotation', 'left/right', 'movement/imagery', 'fist/foot']

MODELS_DIR  = './pretrained'
FIGS_DIR = './figure'
os.makedirs(FIGS_DIR, exist_ok=True)


if __name__ == '__main__':
    # Prepare pretrained model
    pretrained = torch.load(os.path.join(MODELS_DIR, 'dino.pt'))

    # Prepare data
    downstream_datamodule = {
        case: DownstreamDataModule(case, DOWNSTREAM_BATCH_SIZE, DOWNSTREAM_NUM_STEPS)
        for case in TASKS
    }
    for case in TASKS:
        downstream_datamodule[case].setup()
    
    # Evaluate performance of downstream tasks
    performance = pd.DataFrame()
    for case in TASKS:
        num_classes = len(EVENT_ID) if case == 'annotation' else 2
        classifier = KNNClassifier(pretrained, num_classes=num_classes, device='cuda')
        # One-pass training
        print(f'Training/Evaluating for task {case}......')
        for batch in downstream_datamodule[case].train_dataloader():
            classifier.fit(batch)
        # Compute accuracy on testset
        accs = [
            classifier.accuracy(batch)
            for batch in downstream_datamodule[case].test_dataloader()
        ]
        # Append performance summary
        performance = pd.concat([
            performance,
            pd.DataFrame({'task': case, 'accuracy': accs})
        ])
    
    # Visaulization
    performance['accuracy'] *= 100  # rescale to percentage
    ax = sns.boxplot(data=performance, x='accuracy', y='task')
    ax.set(xlabel='accuracy (%)', ylabel='', title='DINO downstream performance')
    plt.savefig(os.path.join(FIGS_DIR, 'dino_downstream.png'))
    plt.show()
