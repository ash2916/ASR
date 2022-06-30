"""
Speech Recognition with Wav2Vec2
================================

**Author**: `Moto Hira <moto@fb.com>`__

This tutorial shows how to perform speech recognition using using
pre-trained models from wav2vec 2.0
[`paper <https://arxiv.org/abs/2006.11477>`__].

"""

######################################################################
# Overview
# --------
#
# The process of speech recognition looks like the following.
#
# 1. Extract the acoustic features from audio waveform
#
# 2. Estimate the class of the acoustic features frame-by-frame
#
# 3. Generate hypothesis from the sequence of the class probabilities
#
# Torchaudio provides easy access to the pre-trained weights and
# associated information, such as the expected sample rate and class
# labels. They are bundled together and available under
# :py:func:`torchaudio.pipelines` module.
#


######################################################################
# Preparation
# -----------
#
# First we import the necessary packages, and fetch data that we work on.
#

# %matplotlib inline

import torch
import torchaudio


torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPEECH_FILE = "./test.flac"

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
print(model)
waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

with torch.inference_mode():
    emission, _ = model(waveform)

print(emission)
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


decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])

print(transcript)
