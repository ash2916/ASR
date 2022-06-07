import torch
import util.latency as latency
from util import model_utility
from util.model_utility import ModelUtility

useGPU = False
device = torch.device("cuda:0" if (useGPU and torch.cuda.is_available()) else "cpu")
# print(latency.run_model(device, **model_utility.contextnet(device=device)))
latency.compare_all()

# contextnet = ModelUtility(model=model_utility.contextnet(device=device).model, model_name="ContextNet",
# dataset="LibriSpeech", num_epochs=10, device=device)
# contextnet.train()
# contextnet.test()
