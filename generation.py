import os
from model.vit.configuration_vit import ViTConfig
from model.vit.modeling_vit import ViTForIPv6Reconstructing
from model.diffusion.modeling_diffusion import DiffusionPriorNetwork, DiffusionPrior
from model import config
import torch
from model import bit_transform
import math
import ipaddress
from tqdm import tqdm
import multiprocessing
from model.tools import get_file_path

def prefix_modify(prefix_input, noise_mean, noise_std, num_channel=3):
    for i in range(num_channel):
        prefix_input[:, i, :, :] -= noise_mean[i]
        prefix_input[:, i, :, :] /= noise_std[i]
    return prefix_input

def suffix_modify(suffix_input, noise_mean, noise_std, num_channel=3):
    for i in range(num_channel):
        suffix_input[:, i, :, :] *= noise_std[i]
        suffix_input[:, i, :, :] += noise_mean[i]
    return suffix_input

def address_construction(prefix, suffix_image, channels, noise_size):
    sub_dict = {}
    suffix = torch.abs(torch.round(suffix_image)).int().tolist()[0]
    for c in range(channels):
        for w in range(64):
            for h in range(64):
                if suffix[c][w][h] != 0 and suffix[c][w][h] != 1:
                    suffix[c][w][h] = 1

    suffix = bit_transform.inverse_bit_transform(suffix, noise_size)

    for c in range(channels):
        for w in range(64):
            suffix_row = list(map(lambda x: str(x), suffix[c][w]))
            suffix_row = ''.join(suffix_row)
            temp_address = ""
            temp_address += prefix
            temp_address += suffix_row
            temp_address = int(temp_address, 2)
            temp_address = ipaddress.IPv6Address(temp_address)
            if temp_address not in sub_dict:
                sub_dict[temp_address] = 1
            else:
                sub_dict[temp_address] += 1
    return sub_dict


def address_generation(prefix=None, num_samples=25600, top_choose=False, topk=1, channels=3, noise_size=1, device='cpu'):

    prefix_noise_mean = config.prefix_noise1_mean
    prefix_noise_std = config.prefix_noise1_std
    suffix_noise_mean = config.suffix_noise1_mean
    suffix_noise_std = config.suffix_noise1_std

    temp_address = ""
    temp_address += prefix
    for i in range(64):
        temp_address += "0"
    temp_address = int(temp_address, 2)
    temp_address = ipaddress.IPv6Address(temp_address)
    prefix_image, _ = bit_transform.single_bit_transform_prefix(prefix, noise_size)

    prefix_list = [prefix_image]

    prefix_input = torch.tensor(prefix_list, dtype=torch.float32).to(device)
    prefix_input = prefix_modify(prefix_input, prefix_noise_mean, prefix_noise_std, channels)
    prefix_embed = prefix_model.vit(prefix_input)[0][:, 1:, :]
    prefix_embed = torch.reshape(prefix_embed, [prefix_embed.shape[0], -1])

    suffix_embed = diffusion_prior.sample(prefix_embed, num_samples_per_batch=num_samples, top_one=top_choose,
                                          topk=topk)
    suffix_embed = torch.reshape(suffix_embed, [suffix_embed.shape[0], 16, 128])

    batch_size, sequence_length, num_channels = suffix_embed.shape
    height = width = math.floor(sequence_length ** 0.5)
    sequence_output = suffix_embed.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

    suffix_output = suffix_model.decoder(sequence_output)
    suffix_output = suffix_output.clone().detach()
    suffix_output = suffix_modify(suffix_output, suffix_noise_mean, suffix_noise_std, channels)

    candidate_list = {}

    if top_choose:
        num_samples = topk

    pbar = tqdm(total=num_samples)
    pbar.set_description(f"Prefix {temp_address} is constructing")

    for i in range(num_samples):
        suffix_image = suffix_output[i:i + 1, :, :, :].clone().detach().to('cpu')
        pbar.update()
        # mp_res.append(pool.apply_async(address_generation, (prefix, suffix_image, channels, noise_size,)))
        sub_dict = address_construction(prefix, suffix_image, channels, noise_size)
        for temp_address in sub_dict:
            if temp_address not in candidate_list:
                candidate_list[temp_address] = sub_dict[temp_address]
            else:
                candidate_list[temp_address] += sub_dict[temp_address]

    pbar.close()
    return candidate_list


if __name__ == "__main__":
    print("please input the 64-bit prefix")
    device = 'cpu'

    topk = 1
    channels = 3
    noise_size = 1

    prefix_model_path = "model_save/vit/prefix_vit-ae_epoch_29"
    suffix_model_path = "model_save/vit/suffix_vit-ae_epoch_29"
    diffusion_model_path = "model_save/diffusion/diffusion_epoch_49"

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

    diffusion_prior.load_state_dict(
        torch.load(diffusion_model_path, map_location=device))

    prefix_model.eval()
    suffix_model.eval()
    diffusion_prior.eval()

    # ----------------------------------------------------------------------------------------------
    while (True):
        prefix = input()
        num_samples = 16
        top_choose = False

        prefix_noise_mean = config.prefix_noise1_mean
        prefix_noise_std = config.prefix_noise1_std
        suffix_noise_mean = config.suffix_noise1_mean
        suffix_noise_std = config.suffix_noise1_std

        temp_address = ""
        temp_address += prefix
        for i in range(64):
            temp_address += "0"
        temp_address = int(temp_address, 2)
        temp_address = ipaddress.IPv6Address(temp_address)
        print("Prefix is: ", temp_address)

        prefix_image, _ = bit_transform.single_bit_transform_prefix(prefix, noise_size)

        prefix_list = [prefix_image]

        prefix_input = torch.tensor(prefix_list, dtype=torch.float32).to(device)
        prefix_input = prefix_modify(prefix_input, prefix_noise_mean, prefix_noise_std, channels)
        prefix_embed = prefix_model.vit(prefix_input)[0][:, 1:, :]
        prefix_embed = torch.reshape(prefix_embed, [prefix_embed.shape[0], -1])

        suffix_embed = diffusion_prior.sample(prefix_embed, num_samples_per_batch=num_samples, top_one=top_choose,
                                              topk=topk)
        suffix_embed = torch.reshape(suffix_embed, [suffix_embed.shape[0], 16, 128])

        batch_size, sequence_length, num_channels = suffix_embed.shape
        height = width = math.floor(sequence_length ** 0.5)
        sequence_output = suffix_embed.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        suffix_output = suffix_model.decoder(sequence_output)
        suffix_output = suffix_output.clone().detach()
        suffix_output = suffix_modify(suffix_output, suffix_noise_mean, suffix_noise_std, channels)


        if top_choose:
            num_samples = topk



        candidate_dict = address_generation(prefix=prefix, num_samples=num_samples, top_choose=top_choose, topk=topk,
                                            channels=channels, noise_size=noise_size, device=device)

        print("the number of generation is: ", len(candidate_dict))
        print(candidate_dict)
