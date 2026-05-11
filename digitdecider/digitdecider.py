from dataclasses import dataclass
import math

from safetensors.torch import save, load
import torch
import torchvision
import torchvision.transforms as transforms

TRAINING_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 0.1

INPUT_SIZE = 28 * 28  # One input for each pixel in a 28 * 28 image
HIDDEN_LAYER_SIZE = 128  # Chosen based on a little vibes, a little goldilocks
OUTPUT_SIZE = 10  # One output per possible classificatoin (10 digits)


@dataclass
class ModelParams:
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor

    def to_safetensors(self) -> bytes:
        model_state = {
            "w1": self.w1.detach(),
            "b1": self.b1.detach(),
            "w2": self.w2.detach(),
            "b2": self.b2.detach(),
        }
        return save(model_state)

    @staticmethod
    def from_safetensors_no_grad(tensorbytes: bytes):
        model_state = load(tensorbytes)
        return ModelParams(**model_state)

    @staticmethod
    def from_safetensors_with_grad(tensorbytes: bytes):
        model_state = load(tensorbytes)
        for tensor in model_state.values():
            tensor.requires_grad_(True)
        return ModelParams(**model_state)


def make_training_data_loader():
    # Turn images into tensors
    # but first, make random changes rotating/stretching/moving/skewing the
    # images so the model learns a more robust generalisation (mnist is very
    # uniform)
    transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10,  # rotate ±10°
                translate=(0.1, 0.1),  # shift up to 10% in each direction
                scale=(0.85, 1.15),  # scale 85%–115%
                shear=10,  # shear up to 10°
            ),
            transforms.ToTensor(),
        ]
    )
    training_data = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset=training_data, batch_size=TRAINING_BATCH_SIZE, shuffle=True
    )


def make_test_data_loader():
    transform = transforms.ToTensor()
    test_data = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset=test_data, batch_size=TEST_BATCH_SIZE, shuffle=False
    )


def init_model_parameters() -> ModelParams:
    """Initialise tensors holding model parameters

    This being an MLP - the model params we need are weights and biases.
    One weight for each edge coming into a node, one bias for the node itself.

    input shape = [TRAINING_BATCH_SIZE, INPUT_SIZE]
    hidden layer activation shape = [TRAINING_BATCH_SIZE, HIDDEN_LAYER_SIZE]

    by [m, k] @ [k, n] = [m, n]

    weights shape (hidden layer incoming) = [INPUT_SIZE, HIDDEN_LAYER_SIZE]

    ---

    We want to then take [TRAINING_BATCH_SIZE, HIDDEN_LAYER_SIZE] activations and get to [TRAINING_BATCH_SIZE, OUTPUT_SIZE]

    so we want weights shape (hidden layer outgoing) = [HIDDEN_LAYER_SIZE, OUTPUT_SIZE]

    bias shape should always be the number of nodes the edges are going to
    """

    w1 = torch.randn(INPUT_SIZE, HIDDEN_LAYER_SIZE) * math.sqrt(2.0 / INPUT_SIZE)
    b1 = torch.zeros(HIDDEN_LAYER_SIZE)

    w2 = torch.randn(HIDDEN_LAYER_SIZE, OUTPUT_SIZE) * math.sqrt(
        2.0 / HIDDEN_LAYER_SIZE
    )
    b2 = torch.zeros(OUTPUT_SIZE)

    # Tell pytorch to set up computation tracking on these tensors so that
    # autograd can use those operations to work out chain-rule derivatives in
    # the backward pass
    for tensor in (w1, b1, w2, b2):
        tensor.requires_grad_(True)

    return ModelParams(w1, b1, w2, b2)


def forward_pass(inputs: torch.Tensor, model_params: ModelParams) -> torch.Tensor:
    """
    Input comes as a 4d tensor [batch_size, num_channels, rows, cols]
    We don't care about the number of channels, or the spatial location of the pixel
    so we can append all the pixel data into one big buffer:
    [batch_size, channels*rows*cols]
    """
    flattened_inputs = inputs.view(inputs.shape[0], -1)
    hidden_layer_preactivated = flattened_inputs @ model_params.w1 + model_params.b1
    hidden_layer_activations = torch.relu(hidden_layer_preactivated)
    output_layer_logits = hidden_layer_activations @ model_params.w2 + model_params.b2
    return output_layer_logits


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, num_classes] - raw model outputs
    labels: [B] - integer class indices
    returns: scalar tensor
    """
    correct_class_logits = logits[range(logits.shape[0]), labels]  # [B]
    log_sum_exp = torch.logsumexp(logits, dim=1)  # [B]
    losses = log_sum_exp - correct_class_logits  # [B]
    return losses.mean()  # scalar


def run_training_step(
    inputs: torch.Tensor, labels: torch.Tensor, params: ModelParams
) -> float:
    # Run forward pass
    result_logits = forward_pass(inputs, params)

    # Calculate loss
    loss = cross_entropy_loss(result_logits, labels)

    # Run backward pass (calculate gradients)
    loss.backward()

    # Adjust weights using stochastic gradient descent (move by gradient * learning rate)
    with torch.no_grad():
        params.w1 -= LEARNING_RATE * params.w1.grad
        params.b1 -= LEARNING_RATE * params.b1.grad
        params.w2 -= LEARNING_RATE * params.w2.grad
        params.b2 -= LEARNING_RATE * params.b2.grad

        # Zero gradients for next step
        params.w1.grad.zero_()
        params.b1.grad.zero_()
        params.w2.grad.zero_()
        params.b2.grad.zero_()

    return loss.item()


def run_epoch(training_data_loader, params: ModelParams) -> float:
    total_loss = 0.0
    batches_trained = 0

    for inputs, labels in training_data_loader:
        total_loss += run_training_step(inputs, labels, params)
        batches_trained += 1

    return total_loss / batches_trained


def evaluate_model(test_data_loader, params: ModelParams) -> float:
    total_correct = 0
    total_predicted = 0

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            result_logits = forward_pass(inputs, params)
            for i in range(0, inputs.shape[0]):
                expected_result = labels[i]
                actual_result = torch.argmax(result_logits[i])
                if expected_result == actual_result:
                    total_correct += 1
            total_predicted += inputs.shape[0]

    return total_correct / total_predicted
