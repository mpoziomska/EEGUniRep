from typing import Any, Callable, Dict, Tuple, Union
import logging

import numpy as np
import pandas as pd
import mne

from torcheeg.datasets import BaseDataset
from torcheeg.utils import get_random_dir_path

from eegunirep.utils.electrode_utils import CHAN_LIST, CHNAMES_MAPPING

import warnings
from scipy.signal import BadCoefficients, iirnotch, sos2zpk, tf2zpk

log = logging.getLogger('torcheeg')

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class FilterSanityException(Exception):
    pass


class FilterCannotBeConstructedException(FilterSanityException):
    pass

class LowFreqPowerTooHighException(FilterSanityException):
    pass


class DCOffsetUnreasonableException(FilterSanityException):
    pass


class ChannelsError(RuntimeError):
    pass


class SignalTooShortException(Exception):
    pass


class LinePowerTooHighException(FilterSanityException):
    pass

class VicRegDataset(BaseDataset):

    def __init__(self,
                 csv_path: str = './data.csv',
                 online_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 **kwargs):
        
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        # log.info(f"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA DUPA {io_size}")
        # pass all arguments to super class
        params = {
            'csv_path': csv_path,
            'read_fn': None,
            'online_transform': online_transform,
            'offline_transform': None,
            'label_transform': label_transform,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose
        }


        params.update(kwargs)
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    def process_record(self, file: Any = None,
                       offline_transform: Union[None, Callable] = None,
                       read_fn: Union[None, Callable] = None,
                       **kwargs):

        trial_info = file
        file_path = trial_info['file_path']

        # log.info(f"FILE PATH {file_path}")
        edf = mne.io.read_raw_edf(input_fname=file_path, preload=True)

        with mne.utils.use_log_level("error"):
            # ujednolica nazwy elektrod oraz ich kolejność,
            # jeśli nie da rady, bo np. brak wystarczającej ilości
            # elektrod, to rzuca wyjątkiem
            i = 0
            while True:
                try:
                    edf = edf.rename_channels(CHNAMES_MAPPING[i])
                    edf = edf.reorder_channels(CHAN_LIST)
                    break
                except ValueError:
                    i += 1

                if i == len(CHNAMES_MAPPING):
                    raise ValueError(
                        f"channels rename/reordering error, available channels\n{edf.info['ch_names']}"
                        f", required channels: {CHAN_LIST}"
                    )

        edf: mne.io.Raw = edf.set_montage("standard_1020", match_case=False) # Czy to jest potrzebne?
        # TODO: szpital MOR ma inne jednostki !!!
        Fs = edf.info['sfreq']
        f_stop = 100
        f_pass = 80
        
        iir_params = dict(ftype='butter', output='sos', gpass=1, gstop=20)
        
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("error")
                iir_params = mne.filter.construct_iir_filter(iir_params,
                                                            f_pass=f_pass,
                                                            f_stop=f_stop,
                                                            sfreq=Fs,
                                                            btype='lowpass',
                                                            return_copy=False)
        except BadCoefficients:
            raise BadCoefficients(f'Bad filter coefficients for lowpass filter '
                                f'settings: {f_stop}, {f_pass}'
                                f' sampling freq {Fs}')
        
        # self.assert_filter_sanity(iir_params, Fs)
        edf = edf.filter(iir_params=iir_params, l_freq=0, h_freq=0, method='iir')
        edf = edf.resample(sfreq=256)

        eeg_raw_signal = edf.get_data(picks=CHAN_LIST, units='uV', tmin=2*60, tmax=30*60)[:, None]
        eid = file_path.split('/')[-1]
        # log.info(f"{file_path.split('/')}, {eid}")
        record_info = {
                **trial_info, 'Fs': 256,
                'clip_id': eid
            }
        # log.info(f"EEG SIGNAL SHAPE: {eeg_raw_signal.shape}")
        for i in range(1):
            yield {'eeg': eeg_raw_signal, 'key': eid, 'info': record_info}

    def assert_filter_sanity(self, filter, fs):
        # print(filter)
        if filter['output'] == 'ba':
            z, p, k = tf2zpk(filter['b'], filter['a'])
        elif filter['output'] == 'sos':
            z, p, k = sos2zpk(filter['sos'])
        runaway_poles = (np.abs(p) > 1.0)  # bieguny poza kołem jednostkowym są niestabilne apriori
        order = len(p)
        if runaway_poles.any():
            runaway_freqs = np.angle(p) * (fs / (2 * np.pi))
            raise FilterCannotBeConstructedException(f'Filter f{filter} has runaway poles! Freqs: f{runaway_freqs}')
        if order > 40 and filter['output'] == 'sos':  # filtry sos są stabilne numerycznie!
            raise FilterCannotBeConstructedException(f'Filter f{filter} order too high! Order: {order}')

        if order > 9 and filter['output'] == 'ba':  # filtry ba są mniej stabilne numerycznie!
            raise FilterCannotBeConstructedException(f'Filter f{filter} order too high! Order: {order}')
    
    def set_records(self, csv_path: str = './data.csv', **kwargs):
        # read csv
        df = pd.read_csv(csv_path)

        assert 'file_path' in df.columns, 'file_path is required in csv file.'

        # df to a list of dict, each dict is a row
        df_list = df.to_dict('records')

        return df_list

    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index).reshape(len(CHAN_LIST), -1)

        signal = eeg
        label = info

        if self.online_transform:
            try:
                signal = self.online_transform(eeg)
            except TypeError:
                signal = self.online_transform(eeg=eeg)

        if self.label_transform:
            label = self.label_transform(y=info)['y']

        return signal, label, info

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'csv_path': self.csv_path,
                'read_fn': self.read_fn,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })