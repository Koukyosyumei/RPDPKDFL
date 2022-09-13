import os
import pickle
import random

import numpy as np
import torch

from ...attack.confidence import get_alpha, get_pi
from ...attack.reconstruction import (
    reconstruct_all_possible_targets,
    reconstruct_private_data_and_quick_evaluate)
from ...attack.tbi_train import (get_inv_train_fn_ablation_3,
                                 get_inv_train_fn_ptbi, get_inv_train_fn_tbi)
from ...model.model import get_model_class
from ...utils.dataloader import prepare_dataloaders
from ...utils.fedkd_setup import get_fedkd_api
from ...utils.tbi_setup import (setup_tbi_optimizers,
                                setup_training_based_inversion)
from ..evaluation.evaluation import evaluation_full


def attack_fedkd(
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
        is_sensitive_flag,
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
    model_class = get_model_class(model_type)
    input_dim = config_dataset["height"] * config_dataset["width"]
    input_dim = (
        input_dim
        if "channel" not in config_dataset
        else input_dim * config_dataset["channel"]
    )

    # --- Setup Optimizers --- #
    (
        inv_path_list,
        inv,
        inv_optimizer,
        inv_optimizer_finetune,
    ) = setup_training_based_inversion(
        attack_type,
        invmodel_type,
        output_dim,
        client_num,
        inv_lr,
        device,
        config_dataset,
        temp_dir,
        ablation_study,
    )

    # --- Setup transformers --- #
    inv_transform = setup_tbi_optimizers(dataset, config_dataset)

    # --- Setup loss function --- #
    criterion = torch.nn.MSELoss()

    if ablation_study in [0, 1, 3, 4]:
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

    if ablation_study in [3, 4]:
        is_sensitive_flag = None

    if attack_type == "ptbi":
        if ablation_study != 3:
            inv_train = get_inv_train_fn_ptbi(
                client_num,
                is_sensitive_flag,
                local_identities,
                inv_transform,
                return_idx,
                seed,
                batch_size,
                num_workers,
                device,
                inv_tempreature,
                inv_batch_size,
                inv_epoch,
                inv_path_list,
                inv,
                inv_optimizer,
                inv_optimizer_finetune,
                criterion,
                output_dim,
                inv_alpha,
                config_dataset,
                config_attack_nes,
                use_finetune,
                pi,
                inv_pj,
                attack_type,
                id2label,
                output_dir,
                ablation_study,
            )
        else:
            inv_train = get_inv_train_fn_ablation_3(
                client_num,
                local_identities,
                inv_transform,
                return_idx,
                seed,
                batch_size,
                num_workers,
                device,
                inv_tempreature,
                inv_batch_size,
                inv_epoch,
                inv_path_list,
                inv,
                inv_optimizer,
                criterion,
            )
    elif attack_type == "tbi":
        inv_train = get_inv_train_fn_tbi(
            client_num,
            local_identities,
            inv_transform,
            return_idx,
            seed,
            batch_size,
            num_workers,
            device,
            inv_tempreature,
            inv_batch_size,
            inv_epoch,
            inv_path_list,
            inv,
            inv_optimizer,
            criterion,
            output_dim,
            inv_pj,
            pi,
            attack_type,
            id2label,
            output_dir,
            ablation_study,
        )
    else:
        raise NotImplementedError(f"{attack_type} is not supported")

    # --- Run FedKD --- #
    api = get_fedkd_api(
        fedkd_type,
        model_class,
        public_train_dataloader,
        local_train_dataloaders,
        test_dataloader,
        num_classes,
        client_num,
        config_dataset["channel"],
        lr,
        num_communication,
        input_dim,
        device,
        config_fedkd,
        custom_action=inv_train,
        target_celeblities_num=config_dataset["target_celeblities_num"],
    )

    fedkd_result = api.run()
    with open(os.path.join(output_dir, "fedkd_result.pkl"), "wb") as f:
        pickle.dump(fedkd_result, f)

    # --- Attack --- #
    if evaluation_type == "quick":
        result = reconstruct_private_data_and_quick_evaluate(
            attack_type,
            local_identities,
            inv_path_list,
            inv,
            inv_optimizer,
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
            inv_optimizer,
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
