# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/spiral/expert.py ]
#   Synopsis     [ the SPIRAL wrapper ]
#   Author       [ Huawei Noah's Lab ]
#   Copyright    [ Copyleft(c), Huawei Noah's Lab]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import fairseq
from ..interfaces import UpstreamBase


############
# CONSTANT #
############
SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    """
    Extract baseline features from wavforms by torchaudio.compliance.kaldi or torchaudio preprocessor
    Support: spectrogram, fbank, mfcc, mel, linear
    """

    def __init__(self, model_config, **kwargs):
        super().__init__(**kwargs)

        with open(model_config, "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

    def forward(self, wavs):
        # import pdb; pdb.set_trace()
        # output: (batch_size, max_sequence_length_of_batch, hidden_size)
        last_hidden_state = wavs[0]['trfm_feat']
        hidden_states = wavs[0][]
        return {
            "last_hidden_state": padded_feats,
            "hidden_states": [padded_feats],
        }