import os

import torch

from ..utils.tbi_setup import setup_inv_dataloader
from .reconstruction import reconstruct_all_possible_targets


def train_inv_model_tbi(data, device, inv_model, optimizer, criterion):
    x = data[0].to(device)
    y_pred_local = data[2].to(device)
    optimizer.zero_grad()
    x_rec_original = inv_model(y_pred_local.reshape(x.shape[0], -1, 1, 1))
    loss = criterion(x, x_rec_original)
    loss.backward()
    optimizer.step()
    return loss, x, x_rec_original


def train_inv_model_ablation_3(data, device, inv_model, optimizer, criterion):
    x = data[0].to(device)
    y_pred_server = data[1].to(device)
    optimizer.zero_grad()
    x_rec_original = inv_model(y_pred_server.reshape(x.shape[0], -1, 1, 1))
    loss = criterion(x, x_rec_original)
    loss.backward()
    optimizer.step()
    return loss, x, x_rec_original


def train_inv_model_bw(data, device, inv_model, optimizer, criterion):
    x = data[0].to(device)
    y_pred_server = data[1].to(device)
    y_pred_local = data[2].to(device)
    if len(data) == 4:
        flag = data[3].to(device)
        y_preds_server_and_local = torch.cat([y_pred_server, y_pred_local, flag], dim=1)
    else:
        y_preds_server_and_local = torch.cat([y_pred_server, y_pred_local], dim=1)

    optimizer.zero_grad()
    x_rec_original = inv_model(y_preds_server_and_local.reshape(x.shape[0], -1, 1, 1))
    loss = criterion(x, x_rec_original)
    loss.backward()
    optimizer.step()
    return loss, x, x_rec_original


def train_ptbi_inv_model(
    prediction_dataloader,
    device,
    inv,
    inv_optimizer,
    criterion,
):
    inv_running_loss = 0
    running_size = 0
    for data in prediction_dataloader:
        loss, x, x_rec = train_inv_model_bw(
            data,
            device,
            inv,
            inv_optimizer,
            criterion,
        )
        inv_running_loss += loss.item()
        running_size += x.shape[0]

    return inv_running_loss / running_size, x, x_rec


def get_inv_train_fn_ptbi(
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
):
    def inv_train(api):
        print("start training inversino models")
        for target_client_id in range(client_num):

            def target_client_api(x_):
                return api.clients[target_client_id](x_).detach()

            # --- Prepare Public Dataset --- #
            # target_labels = local_identities[target_client_id]
            target_labels = sum(local_identities, [])
            prediction_dataloader = setup_inv_dataloader(
                target_labels,
                is_sensitive_flag,
                api,
                target_client_api,
                inv_transform,
                return_idx,
                seed,
                batch_size,
                num_workers,
                device,
                inv_tempreature,
                inv_batch_size,
            )

            checkpoint = torch.load(inv_path_list[target_client_id] + ".pth")
            inv.load_state_dict(checkpoint["model"])
            inv_optimizer.load_state_dict(checkpoint["optimizer"])
            # inv_optimizer_finetune.load_state_dict(checkpoint["finetune_optimizer"])

            for i in range(1, inv_epoch + 1):
                (inv_running_loss, x, x_rec) = train_ptbi_inv_model(
                    prediction_dataloader,
                    device,
                    inv,
                    inv_optimizer,
                    criterion,
                )

                print(f"inv epoch={i}, inv loss ", inv_running_loss)

                with open(
                    os.path.join(output_dir, "inv_result.txt"),
                    "a",
                    encoding="utf-8",
                    newline="\n",
                ) as f:
                    f.write(f"{target_client_id}, {i}, {inv_running_loss}\n")

            state = {
                "model": inv.state_dict(),
                "optimizer": inv_optimizer.state_dict(),
                #"finetune_optimizer": inv_optimizer_finetune.state_dict(),
            }
            torch.save(state, inv_path_list[target_client_id] + ".pth")
            # torch.save(state, inv_path_list[target_client_id] + f"_{api.epoch}.pth")

        if api.epoch % 2 == 1:
            print("saving the reconstructed images...")
            reconstruct_all_possible_targets(
                attack_type,
                is_sensitive_flag,
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
                base_name=api.epoch,
            )

    return inv_train


def get_inv_train_fn_tbi(
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
):
    def inv_train(api):
        for target_client_id in range(client_num):

            def target_client_api(x_):
                return api.clients[target_client_id](x_).detach()

            # --- Prepare Public Dataset --- #
            # target_labels = local_identities[target_client_id]
            target_labels = sum(local_identities, [])
            prediction_dataloader = setup_inv_dataloader(
                target_labels,
                None,
                api,
                target_client_api,
                inv_transform,
                return_idx,
                seed,
                batch_size,
                num_workers,
                device,
                inv_tempreature,
                inv_batch_size,
            )

            checkpoint = torch.load(inv_path_list[target_client_id] + ".pth")
            inv.load_state_dict(checkpoint["model"])
            inv_optimizer.load_state_dict(checkpoint["optimizer"])

            for i in range(1, inv_epoch + 1):
                tbi_running_loss = 0
                running_size = 0
                for data in prediction_dataloader:
                    loss, x, x_rec = train_inv_model_tbi(
                        data,
                        device,
                        inv,
                        inv_optimizer,
                        criterion,
                    )
                    tbi_running_loss += loss.item()
                    running_size += x.shape[0]

                tbi_running_loss /= running_size
                print(f"inv epoch={i}, inv loss ", tbi_running_loss)

                with open(
                    os.path.join(output_dir, "inv_result.txt"),
                    "a",
                    encoding="utf-8",
                    newline="\n",
                ) as f:
                    f.write(f"{target_client_id}, {i}, {tbi_running_loss}\n")

            state = {"model": inv.state_dict(), "optimizer": inv_optimizer.state_dict()}
            torch.save(state, inv_path_list[target_client_id] + ".pth")
            # torch.save(state, inv_path_list[target_client_id] + f"_{api.epoch}.pth")

        """
        if api.epoch % 2 == 1:
            print("saving the reconstructed images...")
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
                base_name=api.epoch,
            )
        """

    return inv_train


def get_inv_train_fn_ablation_3(
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
):
    def inv_train(api):
        for target_client_id in range(client_num):

            def target_client_api(x_):
                return api.clients[target_client_id](x_).detach()

            # --- Prepare Public Dataset --- #
            # target_labels = local_identities[target_client_id]
            target_labels = sum(local_identities, [])
            prediction_dataloader = setup_inv_dataloader(
                target_labels,
                api,
                target_client_api,
                inv_transform,
                return_idx,
                seed,
                batch_size,
                num_workers,
                device,
                inv_tempreature,
                inv_batch_size,
            )

            checkpoint = torch.load(inv_path_list[target_client_id] + ".pth")
            inv.load_state_dict(checkpoint["model"])
            inv_optimizer.load_state_dict(checkpoint["optimizer"])

            for i in range(1, inv_epoch + 1):
                running_loss = 0
                for data in prediction_dataloader:
                    loss, x, x_rec = train_inv_model_ablation_3(
                        data,
                        device,
                        inv,
                        inv_optimizer,
                        criterion,
                    )
                    running_loss += loss.item()

                # plot_two_images(x[0], x_rec[0], titles=["ori", "rec"])
                print(
                    f"inv epoch={i}, inv loss ",
                    running_loss / len(prediction_dataloader),
                )

            state = {"model": inv.state_dict(), "optimizer": inv_optimizer.state_dict()}
            torch.save(state, inv_path_list[target_client_id] + ".pth")
            # torch.save(state, inv_path_list[target_client_id] + f"_{api.epoch}.pth")

    return inv_train
