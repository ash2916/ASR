import torch
import util.latency as latency

# useGPU = True
# device = torch.device("cuda:0" if (useGPU and torch.cuda.is_available()) else "cpu")
# latency.jasper_latency(device)
latency.compare_all()
