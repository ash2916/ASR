import warnings
import torch
import torchaudio
from torchaudio.models.conformer import Conformer
from torchaudio.models.wav2letter import Wav2Letter

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
        print(indices)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


model = Wav2Letter()
model.to('cuda')

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
waveform, sample_rate = torchaudio.load('./richman.wav')
waveform = waveform.to('cuda')

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate).unsqueeze(0)
with torch.inference_mode():
    emission = model(waveform)
decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])
print(transcript)
