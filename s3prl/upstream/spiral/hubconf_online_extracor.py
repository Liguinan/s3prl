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


def spiral_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def spiral_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from google drive id
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return spiral_local(
        _urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs
    )


def spiral(refresh=False, *args, **kwargs):
    """
    The default model - Base
        refresh (bool): whether to download ckpt/config again if existed
    """
    return spiral_base(refresh=refresh, *args, **kwargs)


def spiral_base(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
    return spiral_url(refresh=refresh, *args, **kwargs)


def spiral_large(refresh=False, *args, **kwargs):
    """
    The Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt"
    return spiral_url(refresh=refresh, *args, **kwargs)
