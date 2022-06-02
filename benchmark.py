import latency
import torch
from jasper import Jasper
from contextnet import ContextNet

BATCH_SIZE, SEQ_LENGTH, DIM = 3, 12345, 80

device = torch.device('cpu')
inputs = torch.rand(BATCH_SIZE, SEQ_LENGTH, DIM).to(device)  # BxTxD
input_lengths = torch.LongTensor([SEQ_LENGTH, SEQ_LENGTH - 10, SEQ_LENGTH - 20]).to(device)

# Jasper 10x3 Model Test
model = Jasper(num_classes=10, version='10x5', device=device)
print(latency.run_model(model, inputs=inputs, input_lengths=input_lengths))

# Jasper 5x3 Model Test
model = Jasper(num_classes=10, version='5x3', device=device)
print(latency.run_model(model, inputs=inputs, input_lengths=input_lengths))

