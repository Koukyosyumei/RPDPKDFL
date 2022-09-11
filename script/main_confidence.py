import argparse
import json
import os
import random
import string
from datetime import datetime

from ptbi.pipeline.fedkd.pipeline_confidence_gap import confidence_gap_fedkd


def randomname(n):
    randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    return "".join(randlst)


def add_args(parser):

    parser.add_argument("-l", "--alpha", type=float, default=-1, help="alpha")

    parser.add_argument(
        "-p",
        "--path_to_datafolder",
        type=str,
        default="/content/lag",
        help="path to the data folder",
    )

    parser.add_argument(
        "-g", "--random_seed", type=int, default=42, help="seed of random generator"
    )

    parser.add_argument(
        "-m",
        "--model_folder",
        type=str,
        default="/content/drive/MyDrive/results/",
        help="path to the model folder",
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

    with open(os.path.join(parsed_args.model_folder, "args.txt"), "r") as f:
        args = f.read()

    args = json.loads(
        args.replace("'", '"')
        .replace("<lambda>", "")
        .replace("<", '"')
        .replace(">", '"')
        .replace("None", '"None"')
        .replace("False", '"False"')
        .replace("True", '"True"')
        .replace("inf", '"inf"')
    )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id += "_" + randomname(10)
    run_id += f"_{args['dataset']}_{args['fedkd_type']}_{args['evaluation_type']}_{args['client_num']}"
    run_dir = os.path.join(parsed_args.output_folder, run_id)
    os.makedirs(run_dir)

    args["alpha"] = parsed_args.alpha
    args["random_seed"] = parsed_args.random_seed
    with open(os.path.join(run_dir, "args.txt"), "w") as convert_file:
        convert_file.write(str(args))
    args.pop("alpha")
    args.pop("random_seed")

    print("Start experiment ...")
    print("dataset is ", args["dataset"])
    print("#classes is ", args["num_classes"])
    print("#target classes is ", args["config_dataset"]["target_celeblities_num"])

    confidence_gap_fedkd(
        seed=parsed_args.random_seed,
        model_dir=parsed_args.model_folder,
        output_dir=run_dir,
        temp_dir=run_dir,
        **args,
    )
