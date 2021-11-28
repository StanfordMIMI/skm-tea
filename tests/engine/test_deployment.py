import torch.nn as nn

import skm_tea as st


def test_build_deployment_model():
    model = st.build_deployment_model(
        cfg_or_file="download://https://drive.google.com/file/d/1DTSfmaGu2X9CpE5qW52ux63QrIs9L0oa/view?usp=sharing",  # noqa: E501
        weights_file="download://https://drive.google.com/file/d/1no9-COhdT2Ai3yuxXpSYMpE76hbqZTWn/view?usp=sharing",  # noqa: E501
        force_download=True,
    )
    assert isinstance(model, nn.Module)
