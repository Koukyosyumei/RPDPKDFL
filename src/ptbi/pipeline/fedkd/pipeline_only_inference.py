import os
import random

import numpy as np
import torch

from ...attack.confidence import get_alpha, get_pi
from ...attack.reconstruction import (
    reconstruct_all_possible_targets,
    reconstruct_private_data_and_quick_evaluate,
)
from ...model.invmodel import get_invmodel_class
from ...utils.dataloader import prepare_dataloaders
from ..evaluation.evaluation import evaluation_full


def inference(
    fedkd_type="FedGEMS",
    model_type="LM",
    invmodel_type="InvCNN",
    attack_type="ptbi",
    dataset="AT&T",
    client_num=2,
    batch_size=4,
    inv_batch_size=1,
    lr=0.01,
    num_classes=20,
    num_communication=5,
    seed=42,
    num_workers=2,
    inv_epoch=10,
    inv_lr=0.003,
    inv_tempreature=1.0,
    use_finetune=True,
    inv_pj=0.5,
    beta=0.5,
    evaluation_type="quick",
    ablation_study=0,
    config_fedkd=None,
    config_dataset=None,
    config_attack_nes=None,
    output_dir="",
    model_dir="",
    temp_dir="./",
):
    # --- Fix seed --- #
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except:
        print("torch.use_deterministic_algorithms is not available")

    try:
        torch.backends.cudnn.benchmark = False
    except:
        print("torch.backends.cudnn.benchmark is not available")

    return_idx = True

    # --- Setup device --- #
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print("device is ", device)

    # --- Setup DataLoaders --- #
    (
        public_train_dataloader,
        local_train_dataloaders,
        test_dataloader,
        local_identities,
    ) = prepare_dataloaders(
        dataset_name=dataset,
        client_num=client_num,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        num_classes=num_classes,
        **config_dataset,
    )

    output_dim = (
        num_classes
        if fedkd_type != "DSFL"
        else config_dataset["target_celeblities_num"]
    )

    # --- Setup Models --- #
    input_dim = config_dataset["height"] * config_dataset["width"]
    input_dim = (
        input_dim
        if "channel" not in config_dataset
        else input_dim * config_dataset["channel"]
    )

    if ablation_study in [0, 1, 3]:
        inv_alpha = get_alpha(output_dim, inv_pj)
        pi = get_pi(output_dim, inv_alpha)
    elif ablation_study == 2:
        inv_alpha = None
        pi = 1 / output_dim
        inv_pj = 1 / output_dim
    print(f"pi is {pi}")
    print(f"pj is {inv_pj}")

    if fedkd_type == "DSFL":
        id2label = {la: i for i, la in enumerate(np.unique(sum(local_identities, [])))}
    else:
        id2label = {la: la for la in sum(local_identities, [])}

    if attack_type == "ptbi":
        if ablation_study != 3:
            inv = get_invmodel_class(invmodel_type)(
                input_dim=num_classes * 2,
                output_shape=(
                    config_dataset["channel"],
                    config_dataset["height"],
                    config_dataset["width"],
                ),
                channel=config_dataset["channel"],
            ).to(device)
        else:
            inv = get_invmodel_class(invmodel_type)(
                input_dim=num_classes,
                output_shape=(
                    config_dataset["channel"],
                    config_dataset["height"],
                    config_dataset["width"],
                ),
                channel=config_dataset["channel"],
            ).to(device)
    elif attack_type == "tbi":
        inv = get_invmodel_class(invmodel_type)(
            input_dim=num_classes,
            output_shape=(
                config_dataset["channel"],
                config_dataset["height"],
                config_dataset["width"],
            ),
            channel=config_dataset["channel"],
        ).to(device)

    inv_path_list = [os.path.join(model_dir, f"client_{i}") for i in range(client_num)]

    # --- Attack --- #
    if evaluation_type == "quick":
        result = reconstruct_private_data_and_quick_evaluate(
            attack_type,
            local_identities,
            inv_path_list,
            inv,
            None,
            public_train_dataloader,
            local_train_dataloaders,
            dataset,
            output_dim,
            inv_pj,
            id2label,
            client_num,
            return_idx,
            config_dataset,
            output_dir,
            device,
        )
    elif evaluation_type == "full":
        reconstruct_all_possible_targets(
            attack_type,
            local_identities,
            inv_path_list,
            inv,
            None,
            output_dim,
            inv_pj,
            pi,
            id2label,
            client_num,
            output_dir,
            device,
            ablation_study,
            base_name=str(num_communication),
        )
        result = evaluation_full(
            client_num,
            num_classes,
            public_train_dataloader,
            local_train_dataloaders,
            local_identities,
            id2label,
            attack_type,
            output_dir,
            beta=beta,
            epoch=num_communication,
        )
    else:
        raise NotImplementedError(f"{evaluation_type} is not supported.")

    return result
