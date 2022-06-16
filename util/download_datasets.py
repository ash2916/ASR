import torchaudio

test = torchaudio.datasets.LIBRISPEECH("./../datasets/", url="train-clean-100", download=True)
# test = torchaudio.datasets.LIBRISPEECH("./../datasets/", url="train-clean-360", download=True)
# test = torchaudio.datasets.LIBRISPEECH("./../datasets/", url="train-other-500", download=True)
# test = torchaudio.datasets.LIBRISPEECH("./../datasets/", url="test-clean", download=True)
# test = torchaudio.datasets.LIBRISPEECH("./../datasets/", url="test-other", download=True)
# test = torchaudio.datasets.LIBRISPEECH("./../datasets/", url="dev-clean", download=True)
# test = torchaudio.datasets.LIBRISPEECH("./../datasets/", url="dev-other", download=True)
