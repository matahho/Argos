"""src: A Flower / PyTorch app."""

from src.client import Client
from flwr.client import ClientApp
from flwr.common import Context
from src.dataset_utils import get_dataset_for_client

def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]

    train_loader, validation_loader , test_loader = get_dataset_for_client(
        partition_id=partition_id,
        number_of_paritions=num_partitions,
        batch_size=batch_size
    )

    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return Client(train_loader, validation_loader, local_epochs, learning_rate).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
