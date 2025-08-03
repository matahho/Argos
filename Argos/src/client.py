import torch
from flwr.client import NumPyClient
from src.settings import CLASSES_JSON_FILE
from src.dataset import extract_label_mapping
from src.model import get_model, get_weights, set_weights, train, test


class Client(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        number_of_output_classes = len(extract_label_mapping(classes_file=CLASSES_JSON_FILE))
        self.net = get_model(number_of_output_classes=number_of_output_classes)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        number_of_output_classes = len(extract_label_mapping(classes_file=CLASSES_JSON_FILE))
        net = get_model(number_of_output_classes=number_of_output_classes)
        set_weights(net, parameters=parameters)
        training_results = train(
            net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        )
        trained_weights = get_weights(net)

        return trained_weights, len(self.trainloader.dataset), training_results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        accuracy = test(self.net, self.valloader, self.device)
        return 0.0, len(self.valloader.dataset), {"accuracy": accuracy}
