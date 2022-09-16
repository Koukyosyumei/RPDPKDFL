import os

import numpy as np
import torch


def reconstruct_all_possible_targets(
    attack_type,
    local_identities,
    inv,
    output_dim,
    id2label,
    client_num,
    output_dir,
    device,
    base_name="",
):
    target_ids = sum(local_identities, [])

    for celeb_id in target_ids:
        target_label = id2label[celeb_id]
        dummy_pred_local = torch.zeros(1, output_dim).to(device)
        dummy_pred_local[:, target_label] = 1.0
        x_rec = inv(dummy_pred_local.reshape(1, -1, 1, 1))

        np.save(
            os.path.join(
                output_dir,
                f"{base_name}_{target_label}_{attack_type}",
            ),
            x_rec[0].detach().cpu().numpy(),
        )

    return None
