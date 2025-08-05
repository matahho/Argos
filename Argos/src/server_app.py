"""pytorchexample: A Flower / PyTorch app."""
from collections import OrderedDict
from typing import List, Tuple, Union, Optional
import torch
import numpy as np
from flwr.common import Context, Metrics, ndarrays_to_parameters, FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from src.dataset import extract_label_mapping
from src.model import get_model, get_weights
from src.settings import CLASSES_JSON_FILE , CHECKPOINT_PATH



class SaveModelFedAvgStrategy(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        number_of_output_classes = len(extract_label_mapping(classes_file=CLASSES_JSON_FILE))
        model = get_model(number_of_output_classes=number_of_output_classes)

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `list[np.ndarray]` to PyTorch `state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            # Save the model to disk
            torch.save(model.state_dict(), f"{CHECKPOINT_PATH}/aggregated_model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics








# metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize model parameters
    number_of_output_classes = len(extract_label_mapping(classes_file=CLASSES_JSON_FILE))
    model = get_model(number_of_output_classes=number_of_output_classes)

    ndarrays = get_weights(model)

    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = SaveModelFedAvgStrategy(
        fraction_fit=0.3,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
