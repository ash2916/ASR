import torch
import torch.utils.benchmark as benchmark

from util import model_utility


@torch.no_grad()
def run_inference(model, *argv):
    return model.forward(*argv)


def run_model(device, model_dict):
    num_repeats = 15
    num_threads = torch.get_num_threads()

    globals_dict = {}
    stmt = "run_inference("
    for key, value in model_dict.items():
        globals_dict[key] = value
        stmt = stmt + key + ", "
    model = globals_dict.get('model')
    model.to(device)
    model.eval()
    globals_dict['model'] = model

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
        for num_threads in [1, 4, 16]:
            results.append(benchmark.Timer(
                stmt="run_model(device, model_dict)",
                setup='from util.latency import run_model',
                globals={"device": device, "model_dict": model_utility.contextnet(device=device)},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='ContextNet',
            ).blocked_autorange(min_run_time=1))
            if device == "cuda:0":
                torch.cuda.empty_cache()

    compare = benchmark.Compare(results)
    compare.print()
