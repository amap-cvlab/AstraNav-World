import numpy as np
import argparse
import torch
import habitat
import habitat.config
import habitat.config.default
from habitat.datasets import make_dataset
from habitat import Env

from ovon.dataset import OVONDatasetV1
# from ovon.task.simulator import OVONHabitatConfig
from omegaconf import DictConfig, OmegaConf
from ovon.config import OVONDistanceToGoalConfig
from habitat import Env

from agent.waypoint_agent_ovon import evaluate_agent_ovon


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--split-num",
        type=int,
        required=True,
        help="chunks of evluation"
    )

    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
        help="chunks ID of evluation"

    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="location of model weights"

    )

    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="location to save results"

    )
    from wan.future_frames import add_future_frame_arguments
    add_future_frame_arguments(parser)
    args = parser.parse_args()
    if args.predict_future_frames and not args.wan_model_path:
        parser.error("--predict-future-frames requires --wan-model-path")
    run_exp(**vars(args))


def run_exp(exp_config: str, split_num: str, split_id: str, model_path: str, result_path: str,
            predict_future_frames=False, wan_model_path=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """

    config_all = habitat.get_config(exp_config)
    config = config_all.habitat
    dataset = habitat.make_dataset(config.dataset.type, config=config.dataset)

    np.random.seed(42)
    dataset_split = dataset.get_splits(split_num, allow_uneven_splits=True)[split_id]
    with torch.no_grad():
        evaluate_agent_ovon(config, split_id, dataset_split, model_path, result_path,
                            predict_future_frames=predict_future_frames, wan_model_path=wan_model_path)


if __name__ == "__main__":
    main()
