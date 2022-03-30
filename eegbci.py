"""Utility module for the EEGBCI motor/imagery dataset."""


from multiprocessing.pool import RUN
import numpy as np
import pandas as pd
import mne
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import pytorch_lightning as pl
import sklearn
from typing import Callable, Optional
from augs import UnitScale

# Integer identifier of experimental tasks/events
EVENT_ID = {
    'rest/eye-open': 0,
    'rest/eye-closed': 1,
    'movement/left/fist': 2,
    'movement/right/fist': 3,
    'imagery/left/fist': 4,
    'imagery/right/fist': 5,
    'movement/both/fist': 6,
    'movement/both/foot': 7,
    'imagery/both/fist': 8,
    'imagery/both/foot': 9
}

# Experimental runs and subjects
NUM_SUBJECTS = 109
SUBJECTS_EXCLUDED = [88, 92, 100]  # outliers with a different sample frequency
SUBJECTS = [i for i in range(1, NUM_SUBJECTS+1) if i not in SUBJECTS_EXCLUDED]

NUM_RUNS = 14
RUNS = [i for i in range(1, NUM_RUNS+1)]


def extract_windows(
    raw: mne.io.Raw,
    subject: int,
    run: int,
    event_id: dict,
    num_steps: int = 640,
    baseline_duration: float = 0.2
):
    """
    Return windows/epochs extracted from raw data.

    Arguments
    ---------
    raw: Raw data
    subject: Subject of recording
    run: Experimental run
    event_id: Mapping from tasks/events to integer identifiers
    num_steps: Number of time steps in a window (excluding baseline segment)
    baseline_duration: Duration of baseline for drift correction
    
    Note
    ----
    Baseline segments are not discarded from the returned windows.
    """
    duration = (num_steps - 1) / raw.info['sfreq']
    events, focal_event_id = get_events(raw, run, event_id, duration)
    metadata = pd.DataFrame({
        'start': events[:, 0] / raw.info['sfreq'],  # start time of tasks
        'task': events[:, -1],  # identifier of experimental tasks
        'subject': subject,
        'run': run
    })
    event_mapping = {ind: key for key, ind in event_id.items()}
    epochs = mne.Epochs(
        raw,
        events,
        event_id={event_mapping[ind]: ind for ind in focal_event_id.values()},
        tmin=-baseline_duration,
        tmax=duration,
        metadata=metadata,
        verbose=False
    )
    drop_bad_epochs(
        epochs,
        raw,
        int(duration * raw.info['sfreq']),
        int(baseline_duration * raw.info['sfreq'])
    )
    return epochs


def get_events(
    raw: mne.io.Raw,
    run: int,
    event_id: dict,
    fixed_length_duration: float = 4.0
):
    """
    Return tuple of events and their integer-identifier-mapping according to
    the experimental run.

    Argmuments
    ----------
    raw: Raw data
    run: Experimental run
    event_id: Mapping from tasks/events to integer identifiers
    fixed_length_duration: Duration to create fixed-length events

    Note
    ----
    For runs with alternating experimental tasks, the rest-state periods in
    between will not be considered.
    """
    if run in (1, 2):
        annotation = 'rest/eye-open' if run == 1 else 'rest/eye-closed'
        events = mne.make_fixed_length_events(
            raw,
            id=event_id[annotation],
            duration=fixed_length_duration
        )
        return events, {'T0': event_id[annotation] }
    elif run in (3, 7, 11):
        event_id = {
            'T1': event_id['movement/left/fist'],
            'T2': event_id['movement/right/fist']
        }
        return mne.events_from_annotations(raw, event_id, verbose=False)
    elif run in (4, 8, 12):
        event_id = {
            'T1': event_id['imagery/left/fist'],
            'T2': event_id['imagery/right/fist']
        }
        return mne.events_from_annotations(raw, event_id, verbose=False)
    elif run in (5, 9, 13):
        event_id = {
            'T1': event_id['movement/both/fist'],
            'T2': event_id['movement/both/foot']
        }
        return mne.events_from_annotations(raw, event_id, verbose=False)
    elif run in (6, 10, 14):
        event_id = {
            'T1': event_id['imagery/both/fist'],
            'T2': event_id['imagery/both/foot']
        }
        return mne.events_from_annotations(raw, event_id, verbose=False)
    else:
        raise ValueError('invalid experimental run.')


def drop_bad_epochs(
    epochs: mne.Epochs,
    raw: mne.io.Raw,
    epoch_steps: int,
    baseline_steps: int
) -> None:
    """
    In-place drop epochs that are too short or lack baseline data.

    Arguments
    ---------
    epochs: Epochs to manipulate
    raw: Raw data from which epochs were extracted
    epoch_steps: Number of time steps per epoch
    baseline_steps: Number of time steps of baseline period
    """
    # Drop epochs with fewer steps than the pre-specified number
    mask = (raw.n_times - 1 - epochs.events[:, 0]) < epoch_steps
    epochs.drop(mask, reason='USER: TOO SHORT', verbose=False)
    # Drop epochs which lack baseline data
    mask = (epochs.events[:, 0] - baseline_steps) < 0
    epochs.drop(mask, reason='USER: NO BASELINE', verbose=False)


def download_eegbci(
    subjects: list,
    runs: list,
    dir: str = './'
) -> None:
    """
    Download the EEGBCI motor movement/imagery dat.

    Arguments
    ---------
    subjects: List of subjects of interest
    runs: List of experimental runs for each subject
    dir: Root directory path to the EEGBCI data
    """
    for subject in subjects:
        mne.datasets.eegbci.load_data(subject, runs, path=dir)


def eegbci_epochs_collection(
    subjects: list,
    runs: list,
    dir: str = './',
    **kwargs
) -> list:
    """
    Return extracted windows/epochs from the EEGBCI motor movement/imagery data.

    Arguments
    ---------
    subjects: List of subjects of interest
    runs: List of experimental runs for each subject
    dir: Root directory path to the EEGBCI data
    kwargs: Keyword arguments passed to extract windows
    
    Note
    ----
    Baseline segments are not discarded from the returned windows.
    """
    epochs_list = [
        extract_windows(
            mne.io.read_raw_edf(
                mne.datasets.eegbci.load_data(subject, run, path=dir)[0],
                verbose=False
            ),
            subject,
            run,
            EVENT_ID,
            **kwargs
        ) for subject in subjects for run in runs
    ]
    return epochs_list


class WindowDataset(Dataset):
    """Dataset of extracted windows/epochs."""

    def __init__(
        self,
        windows: mne.Epochs,
        get_label: Callable,
        transform: Callable = None,
        target_transform: Callable = None
    ):
        """
        Arguments
        ---------
        windows: Extracted windows
        get_label: Function to extract label from metadata
        transform: Transformation upon data
        target_transform: Transformation upon labels
        
        Note
        ----
        Baseline segments are discarded from the returned windows.
        """
        super().__init__()
        self.windows = windows
        self.get_label = get_label
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.windows.events)
    
    def __getitem__(self, ind: int):
        window = self.windows[ind]
        data = window.get_data(tmin=0.0)[0]  # baseline segment discarded
        data = torch.from_numpy(data).float()  # convert to float32 tensor
        if self.transform: data = self.transform(data)
        label = self.get_label(window.metadata)
        if self.target_transform: label = self.target_transform(label)
        return data, label


class EEGBCIDataset(ConcatDataset):
    """Dataset of the EEGBCI motor movement/imagery data."""

    def __init__(
        self,
        subjects: list,
        runs: list,
        num_steps: int,
        get_label: Callable,
        transform: Callable = None,
        target_transform: Callable = None,
        **kwargs
    ):
        """
        Arguments
        ---------
        subjects: List of subjects of interest
        runs: List of experimental runs for each subject
        num_steps: Number of time points in each extracted window
        get_label: Function to extract label from metadata
        transform: Transformation upon data
        target_transform: Transformation upon labels
        kwargs: Keyword arguments passed to extract windows
        """
        epochs = eegbci_epochs_collection(
            subjects,
            runs,
            num_steps=num_steps,
            **kwargs
        )
        datasets = [
            WindowDataset(windows, get_label, transform, target_transform)
            for windows in epochs
        ]
        super().__init__(datasets)
        self.metadata = pd.concat([windows.metadata for windows in epochs])


class PretextDataModule(pl.LightningDataModule):
    """EEGBCI motor movement/imagery datamodule for pretext training."""

    def __init__(
        self,
        batch_size: int,
        num_steps: int,
        val_ratio: float = 0.2,
        dir: str = './'
    ):
        """
        Arguments
        ---------
        batch_size: Batch size of training/validation sets
        num_steps: Number of time steps per data sample
        val_ratio: Ratio to split validation set from whole data
        dir: Directory path to store downloaded data
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.val_ratio = val_ratio
        self.data_dir = dir
        self.subjects = SUBJECTS  # all subjects
        self.runs = RUNS  # all runs
        self.get_label = lambda x: -1  # constant (vaccum) get-label function
    
    def prepare_data(self):
        # Download data
        download_eegbci(self.subjects, self.runs, dir=self.data_dir)
    
    def setup(self, stage: Optional[str] = None):
        # Split data into train and validation sets
        dataset = EEGBCIDataset(
            self.subjects,
            self.runs,
            self.num_steps,
            self.get_label,
            dir=self.data_dir,
            transform=UnitScale()
        )
        num_val = int(len(dataset) * self.val_ratio)
        self.trainset, self.valset = torch.utils.data.random_split(
            dataset,
            [len(dataset) - num_val, num_val]
        )
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size)


def runs_classindex_by_case(downstream):
    """
    Return the experimental runs and mapping of relevant events (identifiers)
    to class indices according to the given case of downstream task.
    """
    if downstream == 'annotation':
        runs = RUNS
        class_ind = {ind: ind for ind in EVENT_ID.values()}
    elif downstream == 'left/right':
        runs = [3, 4, 7, 8, 11, 12]
        class_ind = {2: 0, 3: 1, 4: 0, 5: 1}
    elif downstream == 'movement/imagery':
        runs = list(range(3, 14 + 1))
        class_ind = {2: 0, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0, 8: 1, 9: 1}
    elif downstream == 'fist/foot':
        runs = list(range(3, 14 + 1))
        class_ind = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 0, 9: 1}
    elif downstream == 'rest/unrest':
        runs = list(range(1, 14 + 1))
        class_ind = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
    else:
        raise ValueError('downstream task not recognized')
    return runs, class_ind


class DownstreamDataModule(pl.LightningDataModule):
    """EEGBCI motor movement/imagery datamodule for downstream tasks."""

    def __init__(
        self,
        downstream: str,
        batch_size: int,
        num_steps: int,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        dir: str = './'
    ):
        """
        Arguments
        ---------
        downstream: String tag of the downstream task
        batch_size: Batch size of training/validation sets
        num_steps: Number of time steps per data sample
        val_ratio: Ratio to split validation set from whole data
        test_ratio: Ratio to split test set from whole data
        dir: Directory path to store downloaded data
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.data_dir = dir
        self.subjects = SUBJECTS
        self.runs, class_ind = runs_classindex_by_case(downstream)
        # Function to get label for classification
        self.get_label = lambda meta: class_ind[meta['task'].values[0]]
        # Function to get annotation for stratified data splitting
        self.get_annotations = lambda meta: meta['task'].values
    
    def prepare_data(self):
        # Download data
        download_eegbci(self.subjects, self.runs, dir=self.data_dir)
    
    def setup(self, stage: Optional[str] = None):
        # Split data into train, validation, and test sets
        dataset = EEGBCIDataset(
            self.subjects,
            self.runs,
            self.num_steps,
            self.get_label,
            dir=self.data_dir,
            transform=UnitScale()
        )
        num_val = int(len(dataset) * self.val_ratio)
        num_test = int(len(dataset) * self.test_ratio)
        # Splitting is done in a stratified fashion based on annotations
        annts = self.get_annotations(dataset.metadata)
        train_val_inds, test_inds = sklearn.model_selection.train_test_split(
            np.arange(len(dataset)),
            test_size=num_test,
            stratify=annts
        )
        train_inds, val_inds = sklearn.model_selection.train_test_split(
            train_val_inds,
            test_size=num_val,
            stratify=annts[train_val_inds]
        )
        if stage == 'fit' or stage is None:
            self.trainset = Subset(dataset, train_inds)
            self.valset = Subset(dataset, val_inds)
        if stage == 'test' or stage is None:
            self.testset = Subset(dataset, test_inds)
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size)


def main():
    """Empty main."""
    return


if __name__ == '__main__':
    main()
