
import os
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info

from openspeech.tokenizers import TOKENIZER_REGISTRY
from openspeech.datasets import DATA_MODULE_REGISTRY
from openspeech.dataclass.initialize import hydra_train_init
from openspeech.models import MODEL_REGISTRY
from openspeech.utils import parse_configs, get_pl_trainer


@hydra.main(config_path=os.path.join("openspeech", "configs"), config_name="train")
def hydra_main(configs: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(configs))
    pl.seed_everything(configs.trainer.seed)

    logger, num_devices = parse_configs(configs)
    tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)
    model = MODEL_REGISTRY[configs.model.model_name]
    model = model.load_from_checkpoint(configs.eval.checkpoint_path, configs=configs, tokenizer=tokenizer)
    device = torch.device('cuda')
    model.to(device)
    print(model([]))


if __name__ == '__main__':
    hydra_main()
