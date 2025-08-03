
from flwr.client import NumPyClient

from src.dataset_utils import extract_label_mapping
from src.model_manager import ModelManager
from src.settings import CLASSES_JSON_FILE
from src.utils import device_allocation


class Client(NumPyClient):
    def __init__(self, train_loader, validation_loader, local_epochs, learning_rate):
        number_of_output_classes = len(extract_label_mapping(classes_file=CLASSES_JSON_FILE))
        self.model_manger = ModelManager(number_of_output_classes=number_of_output_classes)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device_allocation()

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        model = self.model_manger.get_model()
        self.model_manger.set_weights(model=model, parameters=parameters)
        training_results = self.model_manger.training(
            model=model,
            train_loader=self.train_loader,
            validation_loader=self.validation_loader,
            epochs=self.local_epochs,
            learning_rate=self.learning_rate
        )
        trained_weights = self.model_manger.get_weights(model=model)

        return trained_weights, len(self.train_loader.dataset), training_results

    def evaluate(self, parameters, config):
        model = self.model_manger.get_model()
        self.model_manger.set_weights(model=model, parameters=parameters)
        accuracy = self.model_manger.test(model=model, test_loader=self.validation_loader)
        eval_loss = 0.0
        return eval_loss, len(self.valloader.dataset), {"accuracy": accuracy} # TODO: return the loss of validation
