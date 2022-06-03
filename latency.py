import torch
import torch.utils.benchmark as benchmark

from jasper import Jasper
from contextnet import ContextNet
from las.las_model import Listener
from quartznet.model import QuartzNet
from conformer.model import Conformer


@torch.no_grad()
def run_inference(model, *argv):
    return model.forward(*argv)


def run_model(model, **kwargs):
    num_repeats = 15
    num_threads = 5

    device = torch.device('cpu')
    model.to(device)
    model.eval()

    globals_dict = {"model": model}
    stmt = "run_inference(model"
    for key, value in kwargs.items():
        globals_dict[key] = value
        stmt = stmt + ", " + key
    stmt = stmt + ")"

    # https://pytorch.org/docs/stable/_modules/torch/utils/benchmark/utils/common.html#Measurement
    timer = benchmark.Timer(stmt=stmt,
                            setup="from __main__ import run_inference",
                            globals=globals_dict,
                            num_threads=num_threads,
                            label="Latency Measurement",
                            sub_label="torch.utils.benchmark.")

    profile_result = timer.timeit(num_repeats)
    return f"Latency: {profile_result.mean * 1000:.5f} ms"


def main():
    """
    BATCH_SIZE, SEQ_LENGTH, DIM = 3, 12345, 80

    device = torch.device('cpu')
    inputs = torch.rand(BATCH_SIZE, SEQ_LENGTH, DIM).to(device)  # BxTxD
    input_lengths = torch.LongTensor([SEQ_LENGTH, SEQ_LENGTH - 10, SEQ_LENGTH - 20]).to(device)

    # Jasper 10x3 Model Test
    model = Jasper(num_classes=10, version='5x3', device=device)
    print(run_model(model, inputs=inputs, input_lengths=input_lengths))
    """
    """
    BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE, NUM_VOCABS = 3, 500, 80, 10
    device = torch.device('cpu')

    inputs = torch.FloatTensor(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE).to(device)
    input_lengths = torch.IntTensor([500, 450, 350])
    targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
    target_lengths = torch.LongTensor([9, 8, 7])

    model = ContextNet(
        model_size='large',
        num_vocabs=10, )
    print(run_model(model, inputs=inputs, input_lengths=input_lengths, targets=targets, target_lengths=target_lengths))
    """
    """
    device = torch.device('cpu')
    BATCH_SIZE, TIME_STEP, FEATURE = 3, 784, 39
    input_x = torch.FloatTensor(BATCH_SIZE, TIME_STEP, FEATURE).to(device)
    model = Listener(
        input_feature_dim=39,
        listener_hidden_dim=256,
        listener_layer=2,
        rnn_unit='LSTM',
        use_gpu=False)
    print(run_model(model, input_x=input_x))
    """
    """
    device = torch.device('cpu')
    INPUT, WEIGHT, BIAS = 0, 64, 33
    x = torch.FloatTensor(INPUT, WEIGHT, BIAS).to(device)
    model = QuartzNet(n_mels=64, num_classes=28)
    print(run_model(model, x=x))
    """

    batch_size, sequence_length, dim = 3, 1234, 80
    device = torch.device('cpu')

    inputs = torch.rand(batch_size, sequence_length, dim).to(device)
    input_lengths = torch.IntTensor([1234, 1230, 1200])

    model = Conformer(num_classes=10,
                      input_dim=dim,
                      encoder_dim=32,
                      num_encoder_layers=3)
    print(run_model(model, inputs=inputs, input_lengths=input_lengths))


if __name__ == "__main__":
    main()
