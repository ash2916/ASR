import torch
import torch.utils.data
from torch.optim import *

from models.conformer import Conformer
from models.contextnet import ContextNet
from models.jasper import Jasper
from models.quartznet.model import QuartzNet
from util.librispeech_prepare_data import *
import auraloss


class ModelUtility:
    def __init__(self, model_name, model, device, dataset, num_epochs):
        self.model_name = model_name
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss()
        self.optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        if dataset == "LibriSpeech":
            self.train_loader = librispeech_train_loader
            self.test_loader = librispeech_test_loader
            self.validate_loader = librispeech_validate_loader

    def saveModel(self):
        path = "./saved_models/" + self.model_name + ".pth"
        torch.save(self.model.state_dict(), path)

    def train(self):
        best_accuracy = 0.0

        print("Begin training...")
        for epoch in range(1, self.num_epochs + 1):
            running_train_loss = 0.0
            running_accuracy = 0.0
            running_vall_loss = 0.0
            total = 0

            # Training Loop
            for data in self.train_loader:
                # for data in enumerate(train_loader, 0):
                inputs, outputs = data  # get the input and real species as outputs; data is a list of [inputs, outputs]
                self.optimizer.zero_grad()  # zero the parameter gradients
                predicted_outputs = self.model(inputs)  # predict output from the model
                train_loss = self.loss_fn(predicted_outputs, outputs)  # calculate loss for the predicted output
                train_loss.backward()  # backpropagate the loss
                self.optimizer.step()  # adjust parameters based on the calculated gradients
                running_train_loss += train_loss.item()  # track the loss value

            # Calculate training loss value
            train_loss_value = running_train_loss / len(self.train_loader)

            # Validation Loop
            with torch.no_grad():
                self.model.eval()
                for data in self.validate_loader:
                    inputs, outputs = data
                    predicted_outputs = self.model(inputs)
                    val_loss = self.loss_fn(predicted_outputs, outputs)

                    # The label with the highest value will be our prediction
                    _, predicted = torch.max(predicted_outputs, 1)
                    running_vall_loss += val_loss.item()
                    total += outputs.size(0)
                    running_accuracy += (predicted == outputs).sum().item()

                    # Calculate validation loss value
            val_loss_value = running_vall_loss / len(self.validate_loader)

            # Calculate accuracy as the number of correct predictions in the validation batch divided by the total
            # number of predictions done.
            accuracy = (100 * running_accuracy / total)

            # Save the model if the accuracy is the best
            if accuracy > best_accuracy:
                self.saveModel()
                best_accuracy = accuracy

                # Print the statistics of the epoch
            print('Completed training batch', epoch, 'Training Loss is: %.4f' % train_loss_value,
                  'Validation Loss is: %.4f' % val_loss_value, 'Accuracy is %d %%' % accuracy)

    def test(self):
        input_size = list(torch.Tensor(librispeech_test_data.to_numpy()).shape)[1]
        output_size = librispeech_test_data.__len__()

        # Load the model that we saved at the end of the training loop
        model = self.model(input_size, output_size)
        path = "./saved_models/" + self.model_name + ".pth"
        model.load_state_dict(torch.load(path))

        running_accuracy = 0
        total = 0

        with torch.no_grad():
            for data in self.test_loader:
                inputs, outputs = data
                outputs = outputs.to(torch.float32)
                predicted_outputs = model(inputs)
                _, predicted = torch.max(predicted_outputs, 1)
                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()

            print('Accuracy of the model based on the test set is: %d %%' % (100 * running_accuracy / total))


def jasper(device):
    BATCH_SIZE, SEQ_LENGTH, DIM = 3, 12345, 80

    inputs = torch.rand(BATCH_SIZE, SEQ_LENGTH, DIM).to(device)  # BxTxD
    input_lengths = torch.LongTensor([SEQ_LENGTH, SEQ_LENGTH - 10, SEQ_LENGTH - 20]).to(device)

    # Jasper 10x3 Model Test
    model = Jasper(num_classes=10, version='5x3', device=device)
    return {"model": model, "inputs": inputs, "input_lengths": input_lengths}


def contextnet(device):
    BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE, NUM_VOCABS = 3, 500, 80, 10

    inputs = torch.FloatTensor(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE).to(device)
    input_lengths = torch.IntTensor([500, 450, 350]).to(device)
    targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
    target_lengths = torch.LongTensor([9, 8, 7]).to(device)

    model = ContextNet(
        model_size='medium',
        num_vocabs=10, )
    return {"model": model, "inputs": inputs, "input_lengths": input_lengths, "targets": targets,
            "target_lengths": target_lengths}


def quartznet(device):
    batch_size, sequence_length, dim = 3, 12345, 80
    x = torch.FloatTensor(batch_size, sequence_length, dim).to(device)
    model = QuartzNet(n_mels=12345, num_classes=10)
    return {"model": model, "x": x}


def conformer(device):
    batch_size, sequence_length, dim = 3, 12345, 80

    inputs = torch.rand(batch_size, sequence_length, dim).to(device)
    input_lengths = torch.IntTensor([12345, 12300, 12000])

    model = Conformer(num_classes=10,
                      input_dim=dim,
                      encoder_dim=32,
                      num_encoder_layers=3)
    return {"model": model, "inputs": inputs, "input_lengths": input_lengths}
