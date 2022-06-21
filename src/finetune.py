import copy
import time
from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from utils import (
    fit_resnet_to_face_dataset,
    get_params_to_update,
    set_parameter_requires_grad,
)


class Resnet18Finetuner:
    """Fintuner class"""

    def __init__(self, pretrained: bool = True, feature_extract: bool = True):
        # modify the size of strides and padding to fit the output size of last
        # Conv block to 8*8*512
        self._model = fit_resnet_to_face_dataset(resnet18(pretrained=pretrained))

        # If "feature_extract=True", freeze all Conv blocks of Resnet18
        # If "feature_extract=False", make all Conv blocks of Resnet18 trainable
        # So, we can finetune all Conv blocks.
        self.feature_extract = feature_extract
        set_parameter_requires_grad(self._model, feature_extract)

        # Add final layers to train the target data.
        self.add_final_layers()

        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.scheduler = None

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

    def fit(
        self,
        dataloaders: Dict[str:DataLoader],
        num_epochs: int = 25,
        lr: float = 0.001,
        is_inception: bool = False,
    ) -> [resnet18, Tuple[List, List, List]]:
        self._model = self._model.to(self.device)
        params_to_update = get_params_to_update(self._model, self.feature_extract)

        self.optimizer = optim.Adam(params_to_update, lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1
        )

        since = time.time()
        train_loss_history = []
        train_acc_history = []
        test_loss_history = []
        test_acc_history = []

        best_model_wts = copy.deepcopy(self._model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "test"]:
                if phase == "train":
                    self._model.train()  # Set model to training mode
                else:
                    self._model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == "train":
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = self._model(inputs)
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self._model(inputs)
                            loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "train":
                    self.scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects / len(dataloaders[phase].dataset)

                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

                if phase == "train":
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc.cpu().numpy())
                elif phase == "test":
                    test_loss_history.append(epoch_loss)
                    test_acc_history.append(epoch_acc.cpu().numpy())
                # deep copy the model
                if phase == "test" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self._model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best test Acc: {:4f}".format(best_acc))

        # load best model weights
        self._model.load_state_dict(best_model_wts)
        return self.model, (
            train_loss_history,
            train_acc_history,
            test_loss_history,
            test_acc_history,
        )

    @property
    def model(self) -> resnet18:
        return self._model

    @model.setter
    def model(self, model) -> None:
        self._model = model


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
