from typing import List

from torch import nn


def set_parameter_requires_grad(model, feature_extract: bool):
    # Freeze the parameters of Resnet18
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    return model


def get_params_to_update(model, feature_extract: bool) -> List[nn.Parameter]:
    # Gather the parameters to be optimized/updated
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    return params_to_update


def fit_resnet_to_face_dataset(model):
    # modify the size of strides and padding to fit the output size of last
    # Conv block to 8*8*512
    model.layer3[0].downsample[0] = nn.Conv2d(
        128, 256, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=(1, 1)
    )
    model.layer3[0].conv1 = nn.Conv2d(
        128, 256, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=(1, 1)
    )
    model.layer4[0].downsample[0] = nn.Conv2d(
        256, 512, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=(1, 1)
    )
    model.layer4[0].conv1 = nn.Conv2d(
        256, 512, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=(1, 1)
    )
    return model
