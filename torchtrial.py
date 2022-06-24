import torch
from torch import nn
from torch.optim.adamw import AdamW
from torchaudio.models.conformer import Conformer
from torchaudio.models.wav2letter import Wav2Letter
import torch.utils.data
import torchaudio


# Function to save the model
def saveModel():
    path = "./NetModel.pth"
    torch.save(conformer.state_dict(), path)


train_data = torchaudio.datasets.LIBRISPEECH("./datasets/", url="train-clean-100", download=False)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=1,
                                           shuffle=True)
test_data = torchaudio.datasets.LIBRISPEECH("./datasets/", url="test-clean", download=True)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=1)
validate_data = torchaudio.datasets.LIBRISPEECH("./datasets/", url="dev-clean", download=True)
validate_loader = torch.utils.data.DataLoader(validate_data,
                                              batch_size=1)

conformer = Wav2Letter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
conformer.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(conformer.parameters(), lr=0.001, weight_decay=0.0001)


# Training Function
def train(num_epochs):
    best_accuracy = 0.0

    print("Begin training...")
    for epoch in range(1, num_epochs + 1):
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        total = 0

        # Training Loop
        for i, data in enumerate(train_loader, 0):
            inputs, sample_rate, outputs, a, b, c = data
            optimizer.zero_grad()
            predicted_outputs = conformer(inputs)
            optimizer.step()  # adjust parameters based on the calculated gradients

        # Validation Loop
        with torch.no_grad():
            conformer.eval()
            for data in validate_loader:
                inputs, outputs = data
                predicted_outputs = conformer(inputs)

                # The label with the highest value will be our prediction
                _, predicted = torch.max(predicted_outputs, 1)
                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()
        accuracy = (100 * running_accuracy / total)

        # Save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

            # Print the statistics of the epoch
        print('Completed training batch', epoch, 'Accuracy is %d %%' % accuracy)


def test():
    path = "NetModel.pth"
    conformer.load_state_dict(torch.load(path))

    running_accuracy = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            outputs = outputs.to(torch.float32)
            predicted_outputs = conformer(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()

        print('Accuracy of the model based on the test set is: %d %%' % (100 * running_accuracy / total))


num_epochs = 1
train(num_epochs)
print('Finished Training\n')
test()
