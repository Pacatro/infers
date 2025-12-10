import torch
import torch.nn as nn
import torch.optim as optim
import onnxruntime as ort
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

ONNX_PATH = "../../onnx_models/iris_model.onnx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class IrisNet(nn.Module):
    """
    A fully connected neural network for iris classification.

    The model architecture consists of:
    - Linear operations (General Matrix multiplication)
    - ReLU operation (rectified linear unit)
    - Add operation (addition of two tensors)
    """

    def __init__(self, input_size=4, hidden_size=64, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2_left = nn.Linear(hidden_size, 32)
        self.fc2_right = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        left = torch.relu(self.fc2_left(x))
        right = torch.relu(self.fc2_right(x))
        x = torch.add(left, right)
        x = self.fc3(x)
        return x


def run_onnx_inference(test_input_tensor):
    """
    Load ONNX model and run inference with test input tensor.

    Args:
        test_input_tensor: PyTorch tensor with shape (4,) or (1, 4)
    """
    # Convert PyTorch tensor to numpy and ensure correct shape
    if isinstance(test_input_tensor, torch.Tensor):
        input_numpy = test_input_tensor.cpu().numpy()
    else:
        input_numpy = test_input_tensor

    # Ensure shape is (1, 4) for batch processing
    if input_numpy.ndim == 1:
        input_numpy = input_numpy.reshape(1, -1)

    # Load ONNX model
    session = ort.InferenceSession(ONNX_PATH)

    # Get input name
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(None, {input_name: input_numpy})

    # Get logits and prediction
    logits = outputs[0]
    pred = np.argmax(logits, axis=1)[0]

    print("ONNX Inference Results:")
    print(f"Input shape: {input_numpy.shape}")
    print(f"Logits: {logits}")
    print(f"Predicted class: {pred}")

    return logits, pred


def main():
    Path(ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)
    print(f"Using {device}")

    # Load and prepare the iris dataset
    print("Loading iris dataset")
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test).to(device)

    # Print one instance of the test dataset
    test_input = X_test[0]
    test_target = y_test[0]
    print("\nTest dataset instance:")
    print(f"Features: {test_input}")
    print(f"Shape: {test_input.shape}")
    print(f"Target: {test_target}")
    print(f"Target class name: {iris.target_names[test_target]}")
    print()

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False
    )

    model = IrisNet().to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_dataset)
        accuracy = 100.0 * correct / len(test_dataset)
        print(
            f"Epoch {epoch + 1}/{epochs} - Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

    # Export to ONNX
    dummy_input = torch.randn(1, 4, device=device)
    onnx_program = torch.onnx.export(
        model,
        (dummy_input,),
        dynamo=True,
    )
    assert onnx_program is not None
    onnx_program.save(
        ONNX_PATH, include_initializers=True, keep_initializers_as_inputs=False
    )
    print(f"Model saved as {ONNX_PATH}")

    # Run ONNX inference with test input
    print("\nRunning ONNX inference...")
    run_onnx_inference(test_input)


if __name__ == "__main__":
    main()
