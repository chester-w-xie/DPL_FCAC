"""
-------------------------------File info-------------------------
% - File name: augmentations_spec.py
% - Description:
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2022-07-31
%  Copyright (C) PRMI, South China university of technology; 2022
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: chester.w.xie@gmail.com
------------------------------------------------------------------
"""
import torch
import torchaudio

import numpy as np


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / 10.


def sample_level(n):
    # return np.random.uniform(low=0.1, high=n)
    return np.random.uniform(low=1, high=n)


def time_masking(spec, level):
    size = spec.shape[1]
    level = int_parameter(sample_level(level), size / 3)

    masking = torchaudio.transforms.TimeMasking(time_mask_param=level)

    if not torch.is_tensor(spec):
        spec = torch.from_numpy(spec)

    return masking(spec)


def time_stretch(spec, level):
    n_freq = spec.shape[2]
    level = float_parameter(sample_level(level), 1.8) + 0.1
    stretch = torchaudio.transforms.TimeStretch(n_freq=n_freq)

    spec = torch.transpose(spec, dim0=1, dim1=2)
    spec = stretch(spec, level)
    spec = torch.transpose(spec, dim0=1, dim1=2)
    return spec


def frequency_masking(spec, level):
    size = spec.shape[2]
    level = int_parameter(sample_level(level), size / 1)

    masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=level)
    if not torch.is_tensor(spec):
        spec = torch.from_numpy(spec)

    return masking(spec)


def patch_out(spec, level):
    pass

def time_shift_spectrogram(spectrogram, level):
    """ Shift a spectrogram along the time axis in the spectral-domain at random
    """
    # [ch, time_dim, fre_dim]
    nb_cols = spectrogram.shape[1]
    level = int_parameter(sample_level(level), nb_cols)
    nb_shifts = np.random.randint(0, nb_cols)

    return np.roll(spectrogram, level, axis=1)


# Pitch Shift Augmentation

def pitch_shift_spectrogram(spectrogram, level):
    """ Shift a spectrogram along the frequency axis in the spectral-domain at
    random
    """
    nb_cols = spectrogram.shape[2]
    level = int_parameter(sample_level(level), nb_cols)
    max_shifts = nb_cols // level
    nb_shifts = np.random.randint(-max_shifts, max_shifts)

    return np.roll(spectrogram, nb_shifts, axis=2)


augmentations = [time_masking, frequency_masking, time_shift_spectrogram, pitch_shift_spectrogram]
#
augmentations_all = []

if __name__ == '__main__':
    pass
