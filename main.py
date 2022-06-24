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

import os

import IPython
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio

matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.__version__)
print(torchaudio.__version__)
print(device)

SPEECH_FILE = "./richman.wav"

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())


######################################################################
# Model can be constructed as following. This process will automatically
# fetch the pre-trained weights and load it into the model.
#

model = bundle.get_model().to(device)

print(model.__class__)


######################################################################
# Loading data
# ------------
#
# We will use the speech data from `VOiCES
# dataset <https://iqtlabs.github.io/voices/>`__, which is licensed under
# Creative Commos BY 4.0.
#

IPython.display.Audio(SPEECH_FILE)


######################################################################
# To load data, we use :py:func:`torchaudio.load`.
#
# If the sampling rate is different from what the pipeline expects, then
# we can use :py:func:`torchaudio.functional.resample` for resampling.
#
# .. note::
#
#    - :py:func:`torchaudio.functional.resample` works on CUDA tensors as well.
#    - When performing resampling multiple times on the same set of sample rates,
#      using :py:func:`torchaudio.transforms.Resample` might improve the performace.
#

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)


######################################################################
# Extracting acoustic features
# ---------------------------
with torch.inference_mode():
    emission, _ = model(waveform)

######################################################################
# We can see that there are strong indications to certain labels across
# the time line.
#


######################################################################
# Generating transcripts
# ----------------------
#
# From the sequence of label probabilities, now we want to generate
# transcripts. The process to generate hypotheses is often called
# “decoding”.
#
# Decoding is more elaborate than simple classification because
# decoding at certain time step can be affected by surrounding
# observations.
#
# For example, take a word like ``night`` and ``knight``. Even if their
# prior probability distribution are differnt (in typical conversations,
# ``night`` would occur way more often than ``knight``), to accurately
# generate transcripts with ``knight``, such as ``a knight with a sword``,
# the decoding process has to postpone the final decision until it sees
# enough context.
#
# There are many decoding techniques proposed, and they require external
# resources, such as word dictionary and language models.
#
# In this tutorial, for the sake of simplicity, we will perform greedy
# decoding which does not depend on such external components, and simply
# pick up the best hypothesis at each time step. Therefore, the context
# information are not used, and only one transcript can be generated.
#
# We start by defining greedy decoding algorithm.
#


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


######################################################################
# Now create the decoder object and decode the transcript.
#

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])


######################################################################
# Let’s check the result and listen again to the audio.
#

print(transcript)
IPython.display.Audio(SPEECH_FILE)


######################################################################
# The ASR model is fine-tuned using a loss function called Connectionist Temporal Classification (CTC).
# The detail of CTC loss is explained
# `here <https://distill.pub/2017/ctc/>`__. In CTC a blank token (ϵ) is a
# special token which represents a repetition of the previous symbol. In
# decoding, these are simply ignored.
#


######################################################################
# Conclusion
# ----------
#
# In this tutorial, we looked at how to use :py:mod:`torchaudio.pipelines` to
# perform acoustic feature extraction and speech recognition. Constructing
# a model and getting the emission is as short as two lines.
#
# ::
#
#    model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
#    emission = model(waveforms, ...)
#
