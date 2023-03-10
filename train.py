import logging
import warnings
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything

from heat_anomaly.config import get_configurable_parameters
from heat_anomaly.data import get_datamodule
from heat_anomaly.models import get_model
from heat_anomaly.utils.callbacks import LoadModelCallback, get_callbacks
from heat_anomaly.utils.loggers import configure_logger, get_experiment_logger

logger = logging.getLogger("heat_anomaly")


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="cflow", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, default="heat_anomaly/models/cflow/ir_image_h1.yaml", help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    args = parser.parse_args()
    return args


def train():
    """Train an anomaly classification or segmentation model based on a provided configuration file."""
    args = get_args()
    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config = get_configurable_parameters(model_name=args.model, config_path=args.config)
    if config.project.seed:
        seed_everything(config.project.seed)

    datamodule = get_datamodule(config)
    model = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    logger.info("Training the model.")
    trainer.fit(model=model, datamodule=datamodule)

    logger.info("Loading the best model weights.")
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)

    logger.info("Testing the model.")
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
