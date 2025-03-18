from slickconf import Config, Instance, MainConfig
from pydantic import StrictInt, StrictStr


class Model(Config):
    latent: Instance
    model: Instance
    preconditioner: Instance


class Training(Config):
    image_transform: Instance
    optimizer: Instance
    scheduler: Instance | None = None
    criterion: Instance
    weight_decay: float = 0.0

    n_iters: StrictInt
    train_batch_size: StrictInt
    label_dropout: float = 0
    group_size: StrictInt
    sigma_data: float
    ema: float | None = None


class Data(Config):
    image_dataset: str | None = None
    latent_dataset: str | None = None


class Output(Config):
    log_step: StrictInt = 10
    save_step: StrictInt | None = 1000
    output_dir: StrictStr | None = None
    wandb_project: StrictStr | None = None
    wandb_name: StrictStr | None = None


class IMMConfig(MainConfig):
    model: Model
    training: Training
    data: Data
    output: Output
