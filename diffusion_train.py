import torch
from model.vit.configuration_vit import ViTConfig
from model.vit.modeling_vit import ViTForIPv6Reconstructing
from model.diffusion.modeling_diffusion import DiffusionPriorNetwork, DiffusionPrior
from model import config
from model.dataset import IPv6MixDataset
from torch.utils.data import DataLoader
import os, gc

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    device = 'cpu'
    basic_dataset_path = "data/model_input"
    seed = 42
    epoches = 50

    prefix_model_path = "model_save/vit/prefix_vit-ae_epoch_29"
    suffix_model_path = "model_save/vit/suffix_vit-ae_epoch_29"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    file_path_list = []
    for next_path in os.listdir(basic_dataset_path):
        tmp_path = basic_dataset_path + '/' + next_path
        if not os.path.isfile(tmp_path):
            file_path_list.append(tmp_path)

    vit_config = ViTConfig(
        hidden_size=128,
        num_hidden_layers=16,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.5,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=64,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16
    )

    prefix_model = ViTForIPv6Reconstructing(vit_config).to(device)
    suffix_model = ViTForIPv6Reconstructing(vit_config).to(device)

    prefix_model.load_state_dict(torch.load(prefix_model_path, map_location=device))
    suffix_model.load_state_dict(torch.load(suffix_model_path, map_location=device))

    prior_network = DiffusionPriorNetwork(
        dim=2048,
        depth=4,
        dim_head=128,
        heads=8
    ).to(device)

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        image_embed_dim=2048,
        timesteps=100,
        cond_drop_prob=0.2,
    ).to(device)

    param_count = 0
    for param in diffusion_prior.parameters():
        param_count += param.view(-1).size()[0]
    print("diffusion_prior Parameter is: ", param_count)

    modeling_optimizer = torch.optim.AdamW(diffusion_prior.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                                           weight_decay=0.01)

    prefix_model.eval()
    suffix_model.eval()

    print()

    for epoch in range(epoches):

        print(f"Current epoch is {epoch + 1}/{epoches}.")
        index = 1
        for file_path in file_path_list:
            print(f"    Current file is {index}/{len(file_path_list)}.")
            index += 1
            ipv6_dataset = IPv6MixDataset(file_path=file_path, prefix_mean=config.prefix_noise1_mean,
                                          prefix_std=config.prefix_noise1_std, suffix_mean=config.suffix_noise1_mean,
                                          suffix_std=config.suffix_noise1_std, noise_size=1, num_workers=12)

            train_dataloader = DataLoader(dataset=ipv6_dataset, batch_size=128, shuffle=True, num_workers=12)

            train_rd = 0
            train_loss = 0
            diffusion_prior.train()
            for batch_idx, (data, target) in enumerate(train_dataloader):
                prefix = data.to(device)
                suffix = target.to(device)

                prefix_embed = prefix_model.vit(prefix)[0][:, 1:, :]
                prefix_embed = torch.reshape(prefix_embed, [prefix_embed.shape[0], -1])

                suffix_embed = suffix_model.vit(suffix)[0][:, 1:, :]
                suffix_embed = torch.reshape(suffix_embed, [suffix_embed.shape[0], -1])

                modeling_optimizer.zero_grad()

                loss = diffusion_prior(text_embed=prefix_embed, image_embed=suffix_embed)
                loss.backward()

                modeling_optimizer.step()

                train_rd += 1
                train_loss += loss

                if batch_idx % 100 == 0:
                    print("     Current batch: ", batch_idx, ", train loss: ", loss.data)
            print("Average train loss: ", train_loss / train_rd)

            print("---------------------------")

            del ipv6_dataset
            gc.collect()

        if epoch % 5 == 4:
            torch.save(diffusion_prior.state_dict(), f"model_save/diffusion/diffusion_epoch_{epoch}")

