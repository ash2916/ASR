import torch
import torch.utils.benchmark as benchmark

from models.jasper import Jasper
from models.contextnet import ContextNet
from models.quartznet.model import QuartzNet
from models.conformer.model import Conformer


@torch.no_grad()
def run_inference(model, *argv):
    return model.forward(*argv)


def run_model(device, model, **kwargs):
    num_repeats = 15
    num_threads = torch.get_num_threads()

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
                            setup="from util.latency import run_inference",
                            globals=globals_dict,
                            num_threads=num_threads,
                            label="Latency Measurement",
                            sub_label="torch.utils.benchmark.")

    profile_result = timer.timeit(num_repeats)
    return f"Latency: {profile_result.mean * 1000:.5f} ms"


def compare_all():
    # Compare takes a list of measurements which we'll save in results.
    results = []
    devices = [torch.device("cpu"), ]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    for device in devices:
        # label and sub_label are the rows
        # description is the column
        label = 'Latency'
        sub_label = f'{device}'
        for num_threads in [1, 4, 16, 32]:
            results.append(benchmark.Timer(
                stmt='jasper_latency(device)',
                setup='from util.latency import jasper_latency',
                globals={'device': device},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='Jasper',
            ).blocked_autorange(min_run_time=1))
            results.append(benchmark.Timer(
                stmt='contextnet_latency(device)',
                setup='from util.latency import contextnet_latency',
                globals={'device': device},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='ContextNet',
            ).blocked_autorange(min_run_time=1))
            results.append(benchmark.Timer(
                stmt='quartznet_latency(device)',
                setup='from util.latency import quartznet_latency',
                globals={'device': device},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='QuartzNet',
            ).blocked_autorange(min_run_time=1))
            results.append(benchmark.Timer(
                stmt='conformer_latency(device)',
                setup='from util.latency import conformer_latency',
                globals={'device': device},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='Conformer',
            ).blocked_autorange(min_run_time=1))

    compare = benchmark.Compare(results)
    compare.print()


def jasper_latency(device):
    BATCH_SIZE, SEQ_LENGTH, DIM = 3, 12345, 80

    inputs = torch.rand(BATCH_SIZE, SEQ_LENGTH, DIM).to(device)  # BxTxD
    input_lengths = torch.LongTensor([SEQ_LENGTH, SEQ_LENGTH - 10, SEQ_LENGTH - 20]).to(device)

    # Jasper 10x3 Model Test
    model = Jasper(num_classes=10, version='5x3', device=device)
    print("Jasper " + run_model(device, model, inputs=inputs, input_lengths=input_lengths))


def contextnet_latency(device):
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
    return model
    # print("ContextNet " + run_model(device, model, inputs=inputs, input_lengths=input_lengths, targets=targets,
    #  target_lengths=target_lengths))


def quartznet_latency(device):
    batch_size, sequence_length, dim = 3, 12345, 80
    x = torch.FloatTensor(batch_size, sequence_length, dim).to(device)
    model = QuartzNet(n_mels=12345, num_classes=10)
    print("QuartzNet " + run_model(device, model, x=x))


def conformer_latency(device):
    batch_size, sequence_length, dim = 3, 12345, 80

    inputs = torch.rand(batch_size, sequence_length, dim).to(device)
    input_lengths = torch.IntTensor([12345, 12300, 12000])

    model = Conformer(num_classes=10,
                      input_dim=dim,
                      encoder_dim=32,
                      num_encoder_layers=3)
    print("Conformer " + run_model(device, model, inputs=inputs, input_lengths=input_lengths))
