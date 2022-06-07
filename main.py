import torch
import util.latency as latency
from util.model_utility import ModelUtility

useGPU = False
device = torch.device("cuda:0" if (useGPU and torch.cuda.is_available()) else "cpu")
# latency.jasper_latency(device)
# latency.compare_all()

contextnet = ModelUtility(latency.contextnet_latency(), "ContextNet", dataset="LibriSpeech", num_epochs=10, device=device)
contextnet.train()
contextnet.test()