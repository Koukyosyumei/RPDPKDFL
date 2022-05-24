import os

import numpy as np
import torch

from ..pipeline.evaluation.evaluation import evaluate_ssim
from ..utils.utils_data import (
    extract_transformd_dataset_from_dataloader,
    imshow_dataloader,
)
from .confidence import get_alpha, get_pi


def reconstruct_private_data_and_quick_evaluate(
    attack_type,
    local_identities,
    inv_path_list,
    inv,
    inv_optimizer,
    public_dataloader,
    local_dataloaders,
    dataset,
    output_dim,
    inv_pj,
    id2label,
    client_num,
    return_idx,
    config_dataset,
    output_dir,
    device,
):
    ssim_list = {
        f"{attack_type}_ssim_private": [],
        f"{attack_type}_ssim_public": [],
    }
    result = {
        f"{attack_type}_success": 0,
        f"{attack_type}_falut": 0,
    }

    (
        public_dataset_transformed,
        public_dataset_label,
    ) = extract_transformd_dataset_from_dataloader(
        public_dataloader, return_idx=return_idx
    )

    private_dataset_transformed_list = []
    private_dataset_label_list = []
    for i in range(client_num):
        temp_dataset, temp_label = extract_transformd_dataset_from_dataloader(
            local_dataloaders[i], return_idx=return_idx
        )
        private_dataset_transformed_list.append(temp_dataset)
        private_dataset_label_list.append(temp_label)
    private_dataset_transformed = torch.cat(private_dataset_transformed_list)
    private_dataset_label = torch.cat(private_dataset_label_list)

    for target_client_id, target_list in enumerate(local_identities):

        checkpoint = torch.load(inv_path_list[target_client_id])
        inv.load_state_dict(checkpoint["model"])
        inv_optimizer.load_state_dict(checkpoint["optimizer"])

        for celeb_id in target_list:
            target_label = id2label[celeb_id]
            print(celeb_id, target_label)
            # --- Prepare Attack --- #
            print(f"attack client={target_client_id}, label={target_label}")
            imshow_dataloader(
                [public_dataloader, local_dataloaders[target_client_id]],
                target_label,
                dataset=dataset,
            )

            # --- Training-based inversion attak --- #
            inv_alpha = get_alpha(output_dim, inv_pj)
            print("alpha is ", inv_alpha)
            pi = get_pi(output_dim, inv_alpha)

            dummy_pred_server = torch.ones(1, output_dim).to(device) * pi
            dummy_pred_server[:, target_label] = inv_pj
            dummy_pred_local = torch.zeros(1, output_dim).to(device)
            dummy_pred_local[:, target_label] = 1.0
            dummy_preds = torch.cat([dummy_pred_server, dummy_pred_local], dim=1).to(
                device
            )

            if attack_type == "ptbi":
                x_rec = inv(dummy_preds.reshape(1, -1, 1, 1))
            else:
                x_rec = inv(dummy_pred_local.reshape(1, -1, 1, 1))
            (success_flag, falut_flag, ssim_private, ssim_public,) = evaluate_ssim(
                x_rec[0].detach().cpu().numpy().transpose(1, 2, 0),
                private_dataset_transformed,
                public_dataset_transformed,
                private_dataset_label,
                public_dataset_label,
                output_dim,
                target_label,
                channel=config_dataset["channel"],
            )
            np.save(
                os.path.join(
                    output_dir, f"{target_label}_{attack_type}_{success_flag}"
                ),
                x_rec[0].detach().cpu().numpy(),
            )
            if success_flag:
                print("success!")
                result[f"{attack_type}_success"] += 1
                ssim_list[f"{attack_type}_ssim_private"].append(ssim_private)
                ssim_list[f"{attack_type}_ssim_public"].append(ssim_public)
            if falut_flag:
                print("falut!")
                result[f"{attack_type}_falut"] += 1

            # plt.figure(figsize=(1.5, 1.5))
            # plot_img(x_rec[0], config_dataset["channel"])
            # plt.axis("off")

    for k in ssim_list.keys():
        if len(ssim_list[k]) > 0:
            result[k + "_mean"] = np.mean(ssim_list[k])
            result[k + "_std"] = np.std(ssim_list[k])

    return result


def reconstruct_all_possible_targets(
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
    base_name="",
):
    target_ids = sum(local_identities, [])
    for target_client_id in range(client_num):

        checkpoint = torch.load(inv_path_list[target_client_id])
        inv.load_state_dict(checkpoint["model"])
        inv_optimizer.load_state_dict(checkpoint["optimizer"])

        for celeb_id in target_ids:
            target_label = id2label[celeb_id]
            # --- Prepare Attack --- #

            # --- Training-based inversion attak --- #
            # inv_alpha = get_alpha(output_dim, inv_pj)
            # pi = get_pi(output_dim, inv_alpha)

            dummy_pred_server = torch.ones(1, output_dim).to(device) * pi
            dummy_pred_server[:, target_label] = inv_pj
            dummy_pred_local = torch.zeros(1, output_dim).to(device)
            dummy_pred_local[:, target_label] = 1.0
            dummy_preds = torch.cat([dummy_pred_server, dummy_pred_local], dim=1).to(
                device
            )

            if attack_type == "ptbi":
                if ablation_study != 3:
                    x_rec = inv(dummy_preds.reshape(1, -1, 1, 1))
                else:
                    x_rec = inv(dummy_pred_server.reshape(1, -1, 1, 1))
            else:
                x_rec = inv(dummy_pred_local.reshape(1, -1, 1, 1))

            np.save(
                os.path.join(
                    output_dir,
                    f"{base_name}_{target_label}_{target_client_id}_{attack_type}",
                ),
                x_rec[0].detach().cpu().numpy(),
            )

    return None