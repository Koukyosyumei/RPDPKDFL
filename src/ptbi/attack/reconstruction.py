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

    target_labels = [id2label[celeb_id] for celeb_id in target_ids]
    target_labels_batch = np.array_split(target_labels, int(len(target_labels) / 64))

    for label_batch in target_labels_batch:
        label_batch_tensor = torch.eye(output_dim)[label_batch]
        xs_rec = inv(label_batch_tensor.reshape(len(label_batch), -1, 1, 1))
        xs_rec_array = xs_rec.detach().cpu().numpy()

        for i in range(len(label_batch)):
            np.save(
                os.path.join(
                    output_dir,
                    f"{base_name}_{i}_{attack_type}",
                ),
                xs_rec_array[i].detach().cpu().numpy(),
            )

    return None
