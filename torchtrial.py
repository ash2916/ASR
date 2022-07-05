import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
from nltk.corpus import words
import string

from torchaudio.models import Conformer, ConvTasNet
from tqdm import tqdm

######################################################################
# Let’s check if a CUDA GPU is available and select our device. Running
# the network on a GPU will greatly decrease the training/testing runtime.
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchaudio.datasets import SPEECHCOMMANDS
import os


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = torchaudio.datasets.LIBRISPEECH("./datasets/", url="train-clean-100", download=True)
test_set = torchaudio.datasets.LIBRISPEECH("./datasets/", url="test-clean", download=True)
validate_set = torchaudio.datasets.LIBRISPEECH("./datasets/", url="dev-clean", download=True)
waveform, sample_rate, label, speaker_id, _, utterance_number = train_set[0]
word_list = words.words()
labels = [x for x in word_list]
labels += [x for x in list(string.printable)]
labels = sorted(labels)
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)


def label_to_index(word):
    # Return the position of the word in labels
    global labels
    word = word.lower()
    if word not in labels:
        labels.append(word)
        labels = sorted(labels)
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(x) for x in label.split(" ")]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


batch_size = 512

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
validate_loader = torch.utils.data.DataLoader(
    validate_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
running_accuracy = 0.0
total = 0
best_accuracy = -1
train_loss_value = 0.0


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


# model = ConvTasNet()
model = M5(n_input=transformed.shape[0], n_output=len(labels))
model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


n = count_parameters(model)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Calculate training loss value
        data = data.to(device)
        target = target.to(device)
        data = transform(data)
        output = model(data)
        optimizer.zero_grad()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]")

        # update progress bar
        pbar.update(pbar_update)


# Function to save the model
def saveModel(model):
    path = "./Model.pth"
    torch.save(model.state_dict(), path)


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def validate(model):
    # Validation Loop
    global total
    global running_accuracy
    global best_accuracy
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in enumerate(validate_loader):
            data = data.to(device)
            target = target.to(device)
            data = transform(data)
            output = model(data)
        saveModel(model)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # update progress bar
        pbar.update(pbar_update)



log_interval = 20
n_epoch = 1

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        validate(model)
        test(model, epoch)
        scheduler.step()


def predict(tensor):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device)
    tensor = transform(tensor)
    tensor = tensor.unsqueeze(0)
    tensor = model(tensor)
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor


model.load_state_dict(torch.load('./Model.pth'))
model.eval()
waveform, sample_rate, utterance, *_ = train_set[-1]

print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")

for i, (waveform, sample_rate, utterance, *_) in enumerate(train_set):
    output = predict(waveform)
    print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
waveform, sample_rate = torchaudio.load('./four.wav')
waveform = waveform.to(device)
output = predict(waveform)
print(f" Predicted: {output}.")
# def record(seconds=1):
#
#     from google.colab import output as colab_output
#     from base64 import b64decode
#     from io import BytesIO
#     from pydub import AudioSegment
#
#     RECORD = (
#         b"const sleep  = time => new Promise(resolve => setTimeout(resolve, time))\n"
#         b"const b2text = blob => new Promise(resolve => {\n"
#         b"  const reader = new FileReader()\n"
#         b"  reader.onloadend = e => resolve(e.srcElement.result)\n"
#         b"  reader.readAsDataURL(blob)\n"
#         b"})\n"
#         b"var record = time => new Promise(async resolve => {\n"
#         b"  stream = await navigator.mediaDevices.getUserMedia({ audio: true })\n"
#         b"  recorder = new MediaRecorder(stream)\n"
#         b"  chunks = []\n"
#         b"  recorder.ondataavailable = e => chunks.push(e.data)\n"
#         b"  recorder.start()\n"
#         b"  await sleep(time)\n"
#         b"  recorder.onstop = async ()=>{\n"
#         b"    blob = new Blob(chunks)\n"
#         b"    text = await b2text(blob)\n"
#         b"    resolve(text)\n"
#         b"  }\n"
#         b"  recorder.stop()\n"
#         b"})"
#     )
#     RECORD = RECORD.decode("ascii")
#
#     print(f"Recording started for {seconds} seconds.")
#     s = colab_output.eval_js("record(%d)" % (seconds * 1000))
#     print("Recording ended.")
#     b = b64decode(s.split(",")[1])
#
#     fileformat = "wav"
#     filename = f"_audio.{fileformat}"
#     AudioSegment.from_file(BytesIO(b)).export(filename, format=fileformat)
#     return torchaudio.load(filename)
#
#
# # Detect whether notebook runs in google colab
# if "google.colab" in sys.modules:
#     waveform, sample_rate = record()
#     print(f"Predicted: {predict(waveform)}.")
