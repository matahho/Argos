import uuid

import flwr as fl
import torch.optim as optim
from flwr.common import Context
from torch.utils.data import Subset
import torch

from Argos.Dataset_utils import *
from Argos.Model import get_model, train, evaluate
from Argos.Dataset_utils import number_of_classes, get_dataset_for_client, partitioned_dataset_indices
from Argos.utils import device_allocation

batch_size = 32
learning_rate = 0.001
device = device_allocation()

class Client(fl.client.NumPyClient):
    def __init__(
            self,
            model,
            train_dataset,
            eval_dataset,
            device=device,
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer=None,

    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = model.to(device)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x))
        )
        self.eval_loader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x))
        )

        self.device = device
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate) if not optimizer else optimizer

    def get_parameters(self, config):
        return [
            val.cpu().numpy() for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        train(
            model=self.model,
            dataloader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
        )

        return self.get_parameters(config={}), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        avg_loss , accuracy = evaluate(
            model=self.model,
            dataloader=self.eval_loader,
        )
        return avg_loss, len(self.eval_loader), {"accuracy": accuracy}


def new_client(context : Context) -> Client:
    """Create a Flower client representing a single organization."""

    neural_network = get_model(
        num_classes=number_of_classes
    ).to(device)

    partition_id = context.node_config["partition-id"]
    train_dataset , val_dataset ,test_dataset = get_dataset_for_client(
        partition_id=partition_id,
        full_dataset=dataset,
        partitioned_dataset_indices=partitioned_dataset_indices,
    )


    return Client(
        model=neural_network,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        learning_rate=learning_rate,
        batch_size=batch_size,
    ).to_client()
