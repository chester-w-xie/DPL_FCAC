"""
-------------------------------File info-------------------------
% - File name: DatasetsManager_FSC89_Augmix.py
% - Description:
% -
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2022-08-30
%  Copyright (C) PRMI, South China university of technology; 2022
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: chester.w.xie@gmail.com
------------------------------------------------------------------
"""
import argparse
import os
import numpy as np
import random
import pickle
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import torchaudio
import pandas as pd
import json
import augmentations_spec


def build_label_index(label_unique_list):
    label2inds = defaultdict(list)
    num_labels = len(label_unique_list)
    for idxs, label_unique in enumerate(label_unique_list):
        for label in label_unique:

            if label not in label2inds:
                label2inds[label] = []
            label2inds[label].append(idxs)


    return label2inds


def load_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo)
    return data


def wave_to_tfr(audio_path, norm_data_dir):
    waveform, sr = torchaudio.load(audio_path)

    waveform = waveform - waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                              window_type='hanning', num_mel_bins=128, dither=0.0,
                                              frame_shift=10)
    fbank = fbank.view(1, fbank.shape[0], fbank.shape[1])

    if os.path.isfile(norm_data_dir):

        norm_data = torch.load(norm_data_dir)
        fbank = (fbank - norm_data['mean']) / norm_data['std']

    return fbank


def aug(spec, args_):
    """Perform AugMix augmentations and compute mixture.

  Args:
    args_:
    spec: time-fre feature

  Returns:
    mixed: Augmented and mixed image.
  """
    aug_list = augmentations_spec.augmentations

    ws = np.float32(np.random.dirichlet([1] * args_.mixture_width))

    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(spec)
    for i in range(args_.mixture_width):

        spec_aug = spec.clone()
        depth = args_.mixture_depth if args_.mixture_depth > 0 else np.random.randint(1, 4)

        for _ in range(depth):
            op = np.random.choice(aug_list)

            spec_aug = op(spec_aug, args_.aug_severity)

        mix += ws[i] * spec_aug

    mixed = (1 - m) * spec + m * mix
    return mixed


class AugMixDataset(Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, args=None, sub_indexes=None):
        self.dataset = dataset
        self.args = args
        self.sub_indexes = sub_indexes

    def __getitem__(self, i):
        x, y = self.dataset[i]

        if self.args.no_jsd:
            return aug(x, self.args), y
        else:

            spec_tuple = (x,
                          aug(x, self.args),
                          aug(x, self.args)
                          )
            return spec_tuple, y

    def __len__(self):
        return len(self.dataset)


def get_dataset_norm(_dataset, _dataset_name='FSC-89', phase='train'):
    norm_data = {}
    norm_dir = os.path.join(_dataset_name + '_' + phase + '_' + 'norm_data.pt')


    if not os.path.exists(norm_dir):
        c = []
        print("cal_mean ")
        lengsths_sum = 0
        for i, (x, _) in enumerate(torch.utils.data.DataLoader(_dataset, batch_size=1, shuffle=False, num_workers=2)):

            if i == 0:
                print(x.shape)
            lengsths_sum += x.shape[3]
            x = x[0]
            x = x.transpose(1, 2).contiguous().view(-1, x.size(1))
            c.append(x)
        print("average length", lengsths_sum / len(_dataset))
        print("c [0,1]= ", c[0].size(), c[1].size())
        t = torch.cat(c)  # .transpose(2, 3).contiguous()
        print(t.size())
        m = t.mean(0).float().reshape(1, c[0].size(1), 1)
        print("mean", m.size())
        del t

        c = []
        for i, (x, _) in enumerate(torch.utils.data.DataLoader(_dataset, batch_size=1, shuffle=False, num_workers=2)):
            # x, _, _ = trainset[i]
            x = x[0]
            x = x.transpose(1, 2).contiguous().view(-1, x.size(1))
            c.append(x)
        print("c [0,1]= ", c[0].size(), c[1].size())
        t = torch.cat(c)  # .transpose(2, 3).contiguous()
        print(t.size())
        sd = t.std(0).float().reshape(1, c[0].size(1), 1)
        print("sd", sd.size())

        norm_data['mean'] = m
        norm_data['std'] = sd

        torch.save(norm_data, norm_dir)
    else:
        print(f'Normalized data already exists')


class FSC89Dataset(Dataset):
    def __init__(self, _args, phase=None):
        self.phase = phase
        self.data_dir = _args.datapath
        self.meta_dir = _args.metapath
        self.data_type = _args.data_type
        self.norm_data_dir = os.path.join('FSC-89' + '_' + phase + '_' + 'norm_data.pt')

        if self.phase == 'train':
            meta_info = pd.read_csv(os.path.join(self.meta_dir, _args.setup, 'Fsc89-' + str(_args.setup) +
                                                 '-fsci_train.csv'))

        elif self.phase == 'val':
            meta_info = pd.read_csv(os.path.join(self.meta_dir, _args.setup,  'Fsc89-' + str(_args.setup) +
                                                 '-fsci_val.csv'))

        elif self.phase == 'test':
            meta_info = pd.read_csv(os.path.join(self.meta_dir, _args.setup,  'Fsc89-' + str(_args.setup) +
                                                 '-fsci_test.csv'))
        else:
            raise Exception('No such phase {0}, only support train, val and test'.format(phase))

        self.filenames = meta_info['FSD_MIX_SED_filename']
        self.labels = meta_info['label']
        self.audio_source = meta_info['data_folder']
        self.start_time = meta_info['start_time']

        self.labels = np.array(self.labels)

        self.sub_indexes = defaultdict(list)
        target_max = np.max(self.labels)

        for i in range(target_max + 1):
            self.sub_indexes[i] = np.where(self.labels == i)[0]

    def __getitem__(self, index):

        start_sample = int(self.start_time[index] * 44100)
        if self.data_type == 'openl3':
            filename = self.filenames[index].replace('.wav', '_' + str(start_sample) + '.pkl')
            path = os.path.join(self.data_dir, self.data_type, self.audio_source[index], filename)
            out_data = load_data(path)

        elif self.data_type == 'audio':
            filename = self.filenames[index].replace('.wav', '_' + str(start_sample) + '.wav')
            path = os.path.join(self.data_dir, self.data_type, self.audio_source[index], filename)
            out_data = wave_to_tfr(path, self.norm_data_dir)
        else:
            raise Exception('No such phase {0}, only support openl3 and audio'.format(self.data_type))

        label_out = self.labels[index]

        return out_data, label_out

    def __len__(self):
        return len(self.filenames)


def fsc89_dataset_for_fscil(args_):

    label_per_session = [list(np.array(range(args_.base_class)))] + \
                        [list(np.array(range(args_.way)) + args_.way * task_id + args_.base_class)
                         for task_id in range(args_.tasks)]

    dataset_train = FSC89Dataset(args_, phase='train')
    dataset_val = FSC89Dataset(args_, phase='val')
    dataset_test = FSC89Dataset(args_, phase='test')

    train_datasets = []
    test_datasets = []

    all_datasets = {}

    for session_id in range(args_.session):

        train_datasets.append(SubDatasetTrain(dataset_train, label_per_session, args_, session_id))
        test_datasets.append(SubDatasetTest(dataset_test, label_per_session, session_id))

    all_datasets['train'] = train_datasets
    all_datasets['val'] = dataset_val
    all_datasets['test'] = test_datasets

    return all_datasets

def fsc89_dataset_for_fscil_augmix(args_):

    label_per_session = [list(np.array(range(args_.base_class)))] + \
                        [list(np.array(range(args_.way)) + args_.way * task_id + args_.base_class)
                         for task_id in range(args_.tasks)]

    dataset_train_raw = FSC89Dataset(args_, phase='train')
    dataset_val = FSC89Dataset(args_, phase='val')
    dataset_test = FSC89Dataset(args_, phase='test')

    dataset_train = AugMixDataset(dataset=dataset_train_raw, args=args_, sub_indexes=dataset_train_raw.sub_indexes)

    train_datasets = []
    test_datasets = []

    all_datasets = {}

    for session_id in range(args_.session):

        train_datasets.append(SubDatasetTrain(dataset_train, label_per_session, args_, session_id))
        test_datasets.append(SubDatasetTest(dataset_test, label_per_session, session_id))

    all_datasets['train'] = train_datasets
    all_datasets['val'] = dataset_val
    all_datasets['test'] = test_datasets

    return all_datasets


class SubDatasetTrain(Dataset):
    def __init__(self, dataset, sublabels, args__, task_ids):
        self.ds = dataset
        self.indexes = []
        self.sub_indexes = defaultdict(list)
        if task_ids == 0:
            self.targets = []

            sublabel = sublabels[task_ids]

            for label in sublabel:
                self.indexes.extend(dataset.sub_indexes[int(label)])  #
                self.sub_indexes[label] = dataset.sub_indexes[int(label)]  # -

                temp = np.repeat(label, len(dataset.sub_indexes[int(label)]))

                self.targets.extend(temp)
        else:

            sublabel = sublabels[task_ids]

            for label in sublabel:

                shot_sample = random.sample(list(dataset.sub_indexes[int(label)]), args__.shot)

                self.indexes.extend(shot_sample)
                self.sub_indexes[label] = shot_sample

    def __getitem__(self, item):
        return self.ds[self.indexes[item]]

    def __len__(self):
        return len(self.indexes)


class SubDatasetTest(Dataset):
    def __init__(self, dataset, sublabels, task_ids):
        self.ds = dataset
        self.sub_indexes = []

        for task in range(task_ids + 1):
            sublabel = sublabels[task]
            for label in sublabel:
                self.sub_indexes.extend(dataset.sub_indexes[int(label)])

    def __getitem__(self, item):
        return self.ds[self.sub_indexes[item]]

    def __len__(self):
        return len(self.sub_indexes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metapath', type=str, required=True, help='path to FSC-89-meta folder')
    parser.add_argument('--datapath', type=str, required=True, help='path to FSD-MIX-CLIPS_data folder)')
    parser.add_argument('--setup', type=str, required=True, help='mini or huge')

    # dataset setting(class-division, way, shot)
    parser.add_argument('--base_class', type=int, default=59, help='number of base class (default: 60)')
    parser.add_argument('--way', type=int, default=5, help='class number of per task (default: 5)')
    parser.add_argument('--shot', type=int, default=5, help='shot of per class (default: 5)')
    parser.add_argument('--data_type', type=str, required=True, help='audio or openl3)')

    # hyper option
    parser.add_argument('--session', type=int, default=7, metavar='N',
                        help='num. of sessions, including one base session and n incremental sessions (default:10)')

    # AugMix options
    parser.add_argument('--mixture-width', default=3, type=int, help='Number of augmentation chains to '
                                                                     'mix per augmented example')
    parser.add_argument('--mixture-depth', default=-1, type=int,
                        help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
    parser.add_argument('--aug-severity', default=3, type=int, help='Severity of base augmentation operators')
    parser.add_argument('--no-jsd', '-nj', action='store_true', help='Turn off JSD consistency loss.')
    parser.add_argument('--all-ops', '-all', action='store_true', help='Turn on all operations '
                                                                       '(+brightness,contrast,color,sharpness).')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.tasks = args.session - 1
    print(args)

    train_dataset = FSC89Dataset(args, phase='train')
    val_dataset = FSC89Dataset(args, phase='val')
    test_dataset = FSC89Dataset(args, phase='test')

    get_dataset_norm(train_dataset, phase='train')
    get_dataset_norm(val_dataset, phase='val')
    get_dataset_norm(test_dataset, phase='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=48, shuffle=True,
                                               num_workers=10, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=48 * 2, shuffle=False,
                                              num_workers=10, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=48, shuffle=True,
                                             num_workers=32, pin_memory=True)

    train_dataset_augmix = AugMixDataset(dataset=train_dataset, args=args)
    train_augmix_loader = torch.utils.data.DataLoader(train_dataset_augmix, batch_size=48, shuffle=True,
                                                      num_workers=10, pin_memory=True)

    # data_loader = train_loader
    # data_loader = val_loader
    # data_loader = test_loader
    data_loader = train_augmix_loader

    # loop all the batch
    num_epochs = 1

    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(data_loader):
            # - unpack data
            fea_batch, label_batch = batch_data
            # forward backward ,update, etc.
            if (batch_idx + 1) % 2 == 0:
                print(f'epoch {epoch + 1}/{num_epochs},'
                      f'features shape : {fea_batch[0].shape},'
                      f' label: {label_batch}\n'
                      )
    print('done.\n\n\n\n')

    datasets = fsc89_dataset_for_fscil(args)


    i = 1
    trainset_i = datasets['train'][i]
    valset_0 = datasets['val']
    testset_i = datasets['test'][i]
