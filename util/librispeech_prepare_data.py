import torch.utils.data
import torchaudio

librispeech_train_data = torchaudio.datasets.LIBRISPEECH("./datasets/", url="train-clean-100", download=True)
librispeech_train_loader = torch.utils.data.DataLoader(librispeech_train_data,
                                                       batch_size=10,
                                                       shuffle=True)
librispeech_test_data = torchaudio.datasets.LIBRISPEECH("./datasets/", url="test-clean", download=True)
librispeech_test_loader = torch.utils.data.DataLoader(librispeech_test_data,
                                                      batch_size=1)
librispeech_validate_data = torchaudio.datasets.LIBRISPEECH("./datasets/", url="dev-clean", download=True)
librispeech_validate_loader = torch.utils.data.DataLoader(librispeech_validate_data,
                                                          batch_size=1)
