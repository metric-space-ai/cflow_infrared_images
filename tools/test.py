from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer

from heat_anomaly.config import get_configurable_parameters
from heat_anomaly.data import get_datamodule
from heat_anomaly.models import get_model
from heat_anomaly.utils.callbacks import get_callbacks


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="cflow", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--weight_file", type=str, default="weights/model.ckpt")

    args = parser.parse_args()
    return args


def test():
    args = get_args()
    config = get_configurable_parameters(
        model_name=args.model,
        config_path=args.config,
        weight_file=args.weight_file,
    )

    datamodule = get_datamodule(config)
    model = get_model(config)

    callbacks = get_callbacks(config)

    trainer = Trainer(callbacks=callbacks, **config.trainer)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    test()
