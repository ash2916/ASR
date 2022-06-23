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
import torchaudio
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from openspeech.metrics import WordErrorRate, CharacterErrorRate
from pytorch_lightning.utilities import rank_zero_info

from openspeech.data.audio.dataset import SpeechToTextDataset
from openspeech.data.sampler import RandomSampler
from openspeech.data.audio.data_loader import load_dataset, AudioDataLoader
from openspeech.dataclass.initialize import hydra_eval_init
from openspeech.models import MODEL_REGISTRY
from openspeech.tokenizers import TOKENIZER_REGISTRY


logger = logging.getLogger(__name__)


@hydra.main(config_path=os.path.join("openspeech", "configs"), config_name="predict")
def hydra_main(configs: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(configs))

    tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)
    model = MODEL_REGISTRY[configs.model.model_name](configs=configs, tokenizer=tokenizer)
    waveform, sample_rate = torchaudio.load('./../../../richman.wav')
    downsample_resample = torchaudio.transforms.Resample(
        sample_rate, sample_rate, resampling_method='sinc_interpolation')

    down_sampled = downsample_resample(waveform)
    output = model(down_sampled)
    print(output)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    hydra_eval_init()
    hydra_main()
