import torch
import time
import onnxruntime as ort
import numpy as np


def test_matmul_time():
    t1 = torch.rand(1000, 1000)
    t2 = torch.rand(1000, 1000)

    print("Starting")
    start = time.time()
    t1.matmul(t2)
    print("Matmul time (CPU): ", time.time() - start)

    t1 = t1.to("cuda")
    t2 = t2.to("cuda")

    start = time.time()
    t1.matmul(t2)
    print("Matmul time (GPU): ", time.time() - start)
    t1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    t2 = torch.tensor([[[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    t4 = torch.rand(3, 2, 2)
    print(t1.shape)
    print(t2.shape)
    print(t4.shape)
    t3 = t1.matmul(t4)
    print(t3.shape)
    print(t3)


ort_session = ort.InferenceSession("../onnx_models/mnist_fc_model.onnx")
input_array = np.random.randn(1, 1, 28, 28).astype(np.float32)
inputs = {ort_session.get_inputs()[0].name: input_array}
out = ort_session.run(None, inputs)
print(out)
