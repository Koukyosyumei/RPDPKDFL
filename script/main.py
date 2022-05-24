import argparse
import os
from datetime import datetime

from ptbi.config.config import (
    config_base,
    config_dataset,
    config_fedkd,
    config_nessearch,
)
from ptbi.pipeline.pipeline import attack_fedkd


def add_args(parser):
    parser.add_argument(
        "-t",
        "--fedkd_type",
        type=str,
        default="fedgems",
        help="type of FedKD; FedMD, FedGEMS, or FedGEMS",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="LAG",
    )

    parser.add_argument(
        "-a",
        "--attack_type",
        type=str,
        default="ptbi",
    )

    parser.add_argument(
        "-c",
        "--client_num",
        type=int,
        default=10,
    )

    parser.add_argument(
        "-s",
        "--softmax_tempreature",
        type=float,
        default=1.0,
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

    parser.add_argument(
        "-b",
        "--ablation_study",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    args = config_base
    args["dataset"] = parsed_args.dataset
    args["fedkd_type"] = parsed_args.fedkd_type

    args["attack_type"] = parsed_args.attack_type
    args["client_num"] = parsed_args.client_num

    if parsed_args.ablation_study == 0:
        if parsed_args.fedkd_type == "DSFL":
            args["inv_pj"] = 1.5 * (
                1 / config_dataset[args["dataset"]]["target_celeblities_num"]
            )
        else:
            args["inv_pj"] = 1.5 * (1 / args["num_classes"])
    elif parsed_args.ablation_study == 1:
        args["inv_pj"] = 1  # without entropy term
    elif parsed_args.ablation_study == 2:
        print("inv_pj = 1/output_dim")
    elif parsed_args.ablation_study == 3:
        print("use only the global logit")
        if parsed_args.fedkd_type == "DSFL":
            args["inv_pj"] = 1.5 * (
                1 / config_dataset[args["dataset"]]["target_celeblities_num"]
            )
        else:
            args["inv_pj"] = 1.5 * (1 / args["num_classes"])
    else:
        raise ValueError("parsed_args.ablation_study should be 0, 1, 2 or 3.")

    args["ablation_study"] = parsed_args.ablation_study
    args["inv_tempreature"] = parsed_args.softmax_tempreature

    args["config_dataset"] = config_dataset[args["dataset"]]
    args["config_dataset"]["data_folder"] = parsed_args.path_to_datafolder
    args["config_fedkd"] = config_fedkd[args["fedkd_type"]]
    config_nessearch["temperature"] = parsed_args.softmax_tempreature
    args["config_attack_nes"] = config_nessearch

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id += f"_{args['dataset']}_{args['fedkd_type']}_{args['evaluation_type']}_{args['client_num']}"
    run_dir = os.path.join(parsed_args.output_folder, run_id)
    os.makedirs(run_dir)

    with open(os.path.join(run_dir, "args.txt"), "w") as convert_file:
        convert_file.write(str(args))

    result = attack_fedkd(seed=42, output_dir=run_dir, temp_dir=run_dir, **args)
    print(result)

    with open(os.path.join(run_dir, "result.txt"), "w") as convert_file:
        convert_file.write(str(result))
