from torch import nn
from torchvision.models import resnet18

from utils import fit_resnet_to_face_dataset, set_parameter_requires_grad


class Resnet18Finetuner(object):
    """Fintuner class"""

    def __init__(self, pretrained: bool = True, feature_extract: bool = True):
        # modify the size of strides and padding to fit the output size of last
        # Conv block to 8*8*512
        self._model = fit_resnet_to_face_dataset(resnet18(pretrained=pretrained))

        # If "feature_extract=True", freeze all Conv blocks of Resnet18
        # If "feature_extract=False", make all Conv blocks of Resnet18 trainable
        # So, we can finetune all Conv blocks.
        self._feature_extract = feature_extract
        set_parameter_requires_grad(self._model, feature_extract)

    def add_final_layers(self) -> None:
        # Add final layers to train the target data.
        self._model.fc = nn.Linear(self._model.fc.in_features, 100)
        self._model.batch_norm = nn.BatchNorm2d(
            100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self._model.classifier = nn.Softmax()

    def unfreeze_layer(self, layer_name: str) -> None:
        # Make the weights of specific layer trainable
        dtypes = ["torch.float32", "torch.float64", "torch.complex"]
        for state, param in self._model.state_dict().items():
            if layer_name in state and str(param.dtype) in dtypes:
                param.requires_grad = True

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model) -> None:
        self._model = model

    @property
    def feature_extract(self) -> bool:
        return self._feature_extract

    @feature_extract.setter
    def feature_extract(self, feature_extract) -> None:
        self._feature_extract = feature_extract


class Resnet18FintunerForCaseD(Resnet18Finetuner):
    """
    This class inherit Resnet18Finetuner to override add_final_layers method
    """

    def __init__(self, pretrained, feature_extract):
        super(Resnet18FintunerForCaseD, self).__init__(
            pretrained=pretrained, feature_extract=feature_extract
        )

    def add_final_layers(self) -> None:
        # Override for Case D
        self._model.fc = nn.Linear(self._model.fc.in_features, 100)
        self._model.batch_norm = nn.BatchNorm2d(
            100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self._model.fc_2 = nn.Linear(100, 100)
        self._model.batch_norm_2 = nn.BatchNorm2d(
            100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self._model.classifier = nn.Softmax()
