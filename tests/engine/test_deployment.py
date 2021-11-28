import torch.nn as nn

import skm_tea as st


def test_build_deployment_model():
    model = st.build_deployment_model(
        cfg_or_file="download://https://drive.google.com/file/d/1BJc_lidyHyZvkFD4pLDfSH2Gp2gWXPww/view?usp=sharing",  # noqa: E501
        weights_file="download://https://drive.google.com/file/d/1EkSdXtnD_28_pjZeVFD6XtYugU7VM3jo/view?usp=sharing",  # noqa: E501
        force_download=True,
    )
    assert isinstance(model, nn.Module)
