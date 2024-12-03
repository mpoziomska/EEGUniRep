import numpy as np

CHNAMES_MAPPING = [
    {
        "EEG Fp1": "Fp1",
        "EEG Fp2": "Fp2",
        "EEG F7": "F7",
        "EEG F3": "F3",
        "EEG Fz": "Fz",
        "EEG F4": "F4",
        "EEG F8": "F8",
        "EEG T3": "T3",
        "EEG C3": "C3",
        "EEG Cz": "Cz",
        "EEG C4": "C4",
        "EEG T4": "T4",
        "EEG T5": "T5",
        "EEG P3": "P3",
        "EEG Pz": "Pz",
        "EEG P4": "P4",
        "EEG T6": "T6",
        "EEG O1": "O1",
        "EEG O2": "O2",
    },
    {
        "EEG Fp1": "Fp1",
        "EEG Fp2": "Fp2",
        "EEG F7": "F7",
        "EEG F3": "F3",
        "Fz_nowe": "Fz",
        "EEG F4": "F4",
        "EEG F8": "F8",
        "EEG T3": "T3",
        "EEG C3": "C3",
        "EEG Cz": "Cz",
        "EEG C4": "C4",
        "EEG T4": "T4",
        "EEG T5": "T5",
        "EEG P3": "P3",
        "EEG Pz": "Pz",
        "EEG P4": "P4",
        "EEG T6": "T6",
        "EEG O1": "O1",
        "EEG O2": "O2",
    },
    {
        "EEG FP1-REF": "Fp1",
        "EEG FP2-REF": "Fp2",
        "EEG F3-REF": "F3",
        "EEG F4-REF": "F4",
        "EEG C3-REF": "C3",
        "EEG C4-REF": "C4",
        "EEG P3-REF": "P3",
        "EEG P4-REF": "P4",
        "EEG O1-REF": "O1",
        "EEG O2-REF": "O2",
        "EEG F7-REF": "F7",
        "EEG F8-REF": "F8",
        "EEG T3-REF": "T3",
        "EEG T4-REF": "T4",
        "EEG T5-REF": "T5",
        "EEG T6-REF": "T6",
        "EEG A1-REF": "A1",
        "EEG A2-REF": "A2",
        "EEG FZ-REF": "Fz",
        "EEG CZ-REF": "Cz",
        "EEG PZ-REF": "Pz",
    },
    {
        'C3-REF': "C3",
        'C4-REF': "C4",
        'Cz-REF': "Cz",
        'EKG-REF': "EKG",
        'F3-REF': "F3",
        'F4-REF': "F4",
        'F7-REF': "F7",
        'F8-REF': "F8",
        'Fp1-REF': "Fp1",
        'Fp2-REF': "Fp2",
        'Fz-REF': "Fz",
        'O1-REF': "O1",
        'O2-REF': "O2",
        'P3-REF': "P3",
        'P4-REF': "P4",
        'Photic-REF': "Photic",
        'Pz-REF': "Pz",
        'T3-REF': "T3",
        'T4-REF': "T4",
        'T5-REF': "T5",
        'T6-REF': "T6"},

]


CHAN_LIST = list(CHNAMES_MAPPING[0].values())


CHANNEL_POSITION_MATRIX = np.array(
    [
        ["", "Fp1", "", "Fp2", ""],
        ["F7", "F3", "Fz", "F4", "F8"],
        ["T3", "C3", "Cz", "C4", "T4"],
        ["T5", "P3", "Pz", "P4", "T6"],
        ["", "O1", "", "O2", ""],
    ],
    dtype="str",
)

