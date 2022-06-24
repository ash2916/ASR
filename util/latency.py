import torchtrial
import torchtrial.utils.benchmark as benchmark

from util import model_utility


@torch.no_grad()
def run_inference(device, model, *argv):
    model.to(device)
    model.eval()
    return model.forward(*argv)


def get_inference_params(device, model_dict):
    globals_dict = {}
    stmt = "run_inference(device, "
    for key, value in model_dict.items():
        globals_dict[key] = value
        stmt = stmt + key + ", "
    model = globals_dict.get('model')
    model.to(device)
    model.eval()
    globals_dict['model'] = model
    globals_dict['device'] = device

    stmt = stmt + ")"

    return globals_dict, stmt


def get_latency(device, model_dict):
    num_repeats = 15
    num_threads = torch.get_num_threads()

    globals_dict, stmt = get_inference_params(device, model_dict)

    # https://pytorch.org/docs/stable/_modules/torch/utils/benchmark/utils/common.html#Measurement
    timer = benchmark.Timer(stmt=stmt,
                            setup="from util.latency import run_inference",
                            globals=globals_dict,
                            num_threads=num_threads,
                            label="Latency Measurement",
                            sub_label="torch.utils.benchmark.")

    profile_result = timer.timeit(num_repeats)
    return f"Latency: {profile_result.mean * 1000:.5f} ms"


def compare_all(use_gpu=False, use_cpu=False):
    # Compare takes a list of measurements which we'll save in results.
    results = []
    devices = []
    if use_cpu:
        devices.append(torch.device("cpu"))
    if use_gpu and torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    for device in devices:
        # label and sub_label are the rows
        # description is the column
        label = 'Latency'
        sub_label = f'{device}'
        for num_threads in [1, 4, 16, 32]:
            globals_dict, stmt = get_inference_params(device, model_utility.contextnet(device=device))
            results.append(benchmark.Timer(
                stmt=stmt,
                setup='from util.latency import run_inference',
                globals=globals_dict,
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='ContextNet',
            ).blocked_autorange(min_run_time=1))
            globals_dict, stmt = get_inference_params(device, model_utility.quartznet(device=device))
            results.append(benchmark.Timer(
                stmt=stmt,
                setup='from util.latency import run_inference',
                globals=globals_dict,
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='QuartzNet',
            ).blocked_autorange(min_run_time=1))
            globals_dict, stmt = get_inference_params(device, model_utility.conformer(device=device))
            results.append(benchmark.Timer(
                stmt=stmt,
                setup='from util.latency import run_inference',
                globals=globals_dict,
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='Conformer',
            ).blocked_autorange(min_run_time=1))
            globals_dict, stmt = get_inference_params(device, model_utility.jasper(device=device))
            results.append(benchmark.Timer(
                stmt=stmt,
                setup='from util.latency import run_inference',
                globals=globals_dict,
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='Jasper',
            ).blocked_autorange(min_run_time=1))
            torch.cuda.empty_cache()

    compare = benchmark.Compare(results)
    compare.print()
