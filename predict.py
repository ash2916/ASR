# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import hydra
import warnings
import logging

import torch

import torchtrial
import torchaudio
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info

from openspeech.dataclass.initialize import hydra_eval_init
from openspeech.models import MODEL_REGISTRY
from openspeech.tokenizers import TOKENIZER_REGISTRY

logger = logging.getLogger(__name__)


@hydra.main(config_path=os.path.join("openspeech", "configs"), config_name="predict")
def hydra_main(configs: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(configs))

    tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)
    model = MODEL_REGISTRY[configs.model.model_name]
    model = model.load_from_checkpoint(
        './../../../openspeech_models/contextnet/librispeech/gpu-fp16/fbank/warmup/model.ckpt', configs=configs,
        tokenizer=tokenizer)
    model.to('cuda')

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to('cuda')
    waveform, sample_rate = torchaudio.load('./../../../richman.wav')
    waveform = waveform.to('cuda')

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate).unsqueeze(0)

    with torch.inference_mode():
        emission, _ = model(waveform)
    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(emission[0])
    print(transcript)


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    hydra_eval_init()
    hydra_main()
