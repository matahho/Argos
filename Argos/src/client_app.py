"""pytorchexample: A Flower / PyTorch app."""

from flwr.client import ClientApp
from flwr.common import Context
from src.dataset import load_data
from src.client import Client


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return Client(trainloader, valloader, local_epochs, learning_rate).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
