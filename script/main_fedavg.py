import argparse
import os
from datetime import datetime

from ptbi.config.config import config_base, config_dataset, config_gradinvattack
from ptbi.pipeline.fedavg.pipeline_fedavg import attack_fedavg


def add_args(parser):
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="LAG",
    )

    parser.add_argument(
        "-c",
        "--client_num",
        type=int,
        default=10,
    )

    parser.add_argument(
        "-p",
        "--path_to_datafolder",
        type=str,
        default="/content/lag",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="/content/drive/MyDrive/results/",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    args = {}
    args["dataset"] = parsed_args.dataset
    args["client_num"] = parsed_args.client_num
    print(f"client_num is {args['client_num']}")

    args["model_type"] = config_base["model_type"]
    args["num_communication"] = config_base["num_communication"]
    args["batch_size"] = config_base["batch_size"]
    args["lr"] = config_base["lr"]
    args["num_workers"] = config_base["num_workers"]
    args["num_classes"] = config_base["num_classes"]

    args["config_dataset"] = config_dataset[args["dataset"]]
    args["config_dataset"]["data_folder"] = parsed_args.path_to_datafolder
    args["config_gradinvattack"] = config_gradinvattack

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id += f"_{args['dataset']}_FedAVG_{args['client_num']}"
    run_dir = os.path.join(parsed_args.output_folder, run_id)
    os.makedirs(run_dir)

    with open(os.path.join(run_dir, "args.txt"), "w") as convert_file:
        convert_file.write(str(args))

    result = attack_fedavg(seed=42, output_dir=run_dir, **args)
    print(result)

    with open(os.path.join(run_dir, "result.txt"), "w") as convert_file:
        convert_file.write(str(result))
