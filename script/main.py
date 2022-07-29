import argparse
import os
from datetime import datetime

from ptbi.attack import get_pj
from ptbi.config.config import config_base, config_dataset, config_fedkd
from ptbi.pipeline.fedkd.pipeline import attack_fedkd


def add_args(parser):
    parser.add_argument(
        "-t",
        "--fedkd_type",
        type=str,
        default="fedgems",
        help="type of FedKD; FedMD, FedGEMS, or FedGEMS",
    )

    parser.add_argument(
        "-d", "--dataset", type=str, default="LAG", help="type of dataset; LAG or LFW"
    )

    parser.add_argument(
        "-a",
        "--attack_type",
        type=str,
        default="ptbi",
        help="type of attack; ptbi or tbi",
    )

    parser.add_argument("-l", "--alpha", type=float, default=-1, help="alpha")

    parser.add_argument(
        "-c", "--client_num", type=int, default=10, help="number of clients"
    )

    parser.add_argument(
        "-s",
        "--softmax_tempreature",
        type=float,
        default=1.0,
        help="tempreature $\tau$",
    )

    parser.add_argument(
        "-u",
        "--blur_strength",
        type=int,
        default=10,
        help="strength of blur",
    )

    parser.add_argument(
        "-p",
        "--path_to_datafolder",
        type=str,
        default="/content/lag",
        help="path to the data folder",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="/content/drive/MyDrive/results/",
        help="path to the output folder",
    )

    parser.add_argument(
        "-b",
        "--ablation_study",
        type=int,
        default=0,
        help="type of ablation study; 0:normal(Q=p'_{c_i, j}+p'_{s, j}+\alpha H(p'_s)), \
                                      1:without entropy (Q=p'_{c_i, j}+p'_{s, j})\
                                      2:without p'_{s, j} (Q=p'_{c_i, j}+\alpha H(p'_s))\
                                      3:without local logit (Q=p'_{s, j}+\alpha H(p'_s))",
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

    if args["dataset"] == "AT&T":
        args["num_classes"] = 400

    if parsed_args.ablation_study == 0:
        if parsed_args.fedkd_type == "DSFL":
            if parsed_args.alpha < 0:
                args["inv_pj"] = 1.5 * (
                    1 / config_dataset[args["dataset"]]["target_celeblities_num"]
                )
            else:
                args["inv_pj"] = get_pj(
                    config_dataset[args["dataset"]]["target_celeblities_num"],
                    parsed_args.alpha,
                )
        else:
            if parsed_args.alpha < 0:
                args["inv_pj"] = 1.5 * (1 / args["num_classes"])
            else:
                args["inv_pj"] = get_pj(args["num_classes"], parsed_args.alpha)
    elif parsed_args.ablation_study == 1:
        args["inv_pj"] = 1  # without entropy term
    elif parsed_args.ablation_study == 2:
        print("inv_pj = 1/output_dim")
    elif parsed_args.ablation_study == 3:
        print("use only the global logit")
        if parsed_args.fedkd_type == "DSFL":
            if parsed_args.alpha < 0:
                args["inv_pj"] = 1.5 * (
                    1 / config_dataset[args["dataset"]]["target_celeblities_num"]
                )
            else:
                args["inv_pj"] = get_pj(
                    config_dataset[args["dataset"]]["target_celeblities_num"],
                    parsed_args.alpha,
                )
        else:
            if parsed_args.alpha < 0:
                args["inv_pj"] = 1.5 * (1 / args["num_classes"])
            else:
                args["inv_pj"] = get_pj(args["num_classes"], parsed_args.alpha)
    else:
        raise ValueError("parsed_args.ablation_study should be 0, 1, 2 or 3.")

    args["ablation_study"] = parsed_args.ablation_study
    args["inv_tempreature"] = parsed_args.softmax_tempreature

    args["config_dataset"] = config_dataset[args["dataset"]]
    args["config_dataset"]["data_folder"] = parsed_args.path_to_datafolder
    args["config_fedkd"] = config_fedkd[args["fedkd_type"]]

    if args["dataset"] == "AT&T":
        args["config_dataset"]["blur_strength"] = parsed_args.blur_strength

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
