# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/spiral/hubconf.py ]
#   Synopsis     [ the spiral torch hubconf ]
#   Author       [ Huawei Noah's Lab ]
#   Copyright    [ Copyleft(c), Huawei Noah's Lab]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import os

# -------------#
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert

def baseline_local(model_config, *args, **kwargs):
    """
        Baseline feature
            model_config: PATH
    """
    assert os.path.isfile(model_config)
    return _UpstreamExpert(model_config, *args, **kwargs)


def spiral(*args, **kwargs):
    """
        
    """
    kwargs['model_config'] = os.path.join(os.path.dirname(__file__), 'spiral.yaml')
    return baseline_local(*args, **kwargs)