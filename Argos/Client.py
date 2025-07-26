import flwr as fl
import torch.optim as optim
from flwr.common import Context
from Argos.Dataset_utils import *
from Argos.Model import get_model, train, evaluate
from Argos.Dataset_utils import get_dataset_for_client
from Argos.utils import device_allocation
from Argos.settings import DEVICE , CLIENT_LEARNING_RATE , CLIENT_BATCH_SIZE

class Client(fl.client.NumPyClient):
    def __init__(
            self,
            model,
            train_dataset,
            eval_dataset,
            device=DEVICE,
            learning_rate=CLIENT_LEARNING_RATE,
            batch_size=CLIENT_BATCH_SIZE,
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

