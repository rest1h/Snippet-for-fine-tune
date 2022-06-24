import copy
import time
from typing import Dict, Iterable, List, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from utils import get_params_to_update


class Trainer(object):
    def __init__(self, finetuner):
        self._finetuner = finetuner
        self._model = finetuner.model

    def fit(
        self,
        dataloaders: Dict[str, Iterable[DataLoader]],
        num_epochs: int,
        lr: float,
        optimizer,
        criterion,
        device: str,
        is_inception: bool = False,
    ) -> Tuple[resnet18, Tuple[List[float], List[float], List[float], List[float]]]:

        self._model = self._model.to(device)
        params_to_update = get_params_to_update(
            self._model, self._finetuner.feature_extract
        )

        optimizer = optimizer(params_to_update, lr=lr)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1
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
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == "train":
                            outputs, aux_outputs = self._model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self._model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "train":
                    scheduler.step()

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
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
