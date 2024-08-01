import torch
from model import config
from model.dataset import IPv6MixDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from model.vit.modeling_vit import ViTForIPv6Reconstructing
from model.vit.configuration_vit import ViTConfig
import gc, os

if __name__ == "__main__":

    device = 'cpu'
    basic_dataset_path = "data/model_input"
    seed = 42
    epoches = 30

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

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    file_path_list = []
    for next_path in os.listdir(basic_dataset_path):
        tmp_path = basic_dataset_path + '/' + next_path
        if not os.path.isfile(tmp_path):
            file_path_list.append(tmp_path)

    vit_modeling_model = ViTForIPv6Reconstructing(vit_config).to(device)

    param_count = 0
    for param in vit_modeling_model.parameters():
        param_count += param.view(-1).size()[0]
    print("Parameter is: ", param_count)
    modeling_optimizer = torch.optim.AdamW(vit_modeling_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    best_model = ""
    best_loss = 10000000000

    for epoch in range(epoches):
        print(f"Current epoch is {epoch+1}/{epoches}.")
        index = 1
        for file_path in file_path_list:
            print(f"    Current file is {index}/{len(file_path_list)}.")
            index += 1
            ipv6_dataset = IPv6MixDataset(file_path=file_path, prefix_mean=config.prefix_noise1_mean,
                                  prefix_std=config.prefix_noise1_std, suffix_mean=config.suffix_noise1_mean,
                                  suffix_std=config.suffix_noise1_std, noise_size=1, num_workers=12)
            train_size = int(0.8*len(ipv6_dataset))
            test_size = len(ipv6_dataset)-train_size
            train_dataset, test_dataset = torch.utils.data.random_split(ipv6_dataset, [train_size, test_size])

            train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=12)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=128, num_workers=12)

            vit_modeling_model.train()
            train_rd = 0
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_dataloader):
                target = target.to(device)
                modeling_optimizer.zero_grad()
                result = vit_modeling_model(target)
                loss = result.loss

                loss.backward()
                modeling_optimizer.step()

                train_rd += 1
                train_loss += loss

                if batch_idx % 100 == 0:
                    print("     Current batch: ", batch_idx, " ", loss.data)

            print("Average train loss: ", train_loss / train_rd)

            vit_modeling_model.eval()
            valid_loss = 0
            valid_rd = 0
            for batch_idx, (data, target) in enumerate(test_dataloader):
                target = target.to(device)
                with torch.no_grad():
                    result = vit_modeling_model(target)

                loss = result.loss
                valid_loss += loss
                valid_rd += 1

            print("Valid loss: ", valid_loss / valid_rd)

            print("---------------------------")

            if best_loss > valid_loss / valid_rd:
                best_loss = valid_loss / valid_rd
                best_model = f"encoder_epoch_{epoch}"

            del ipv6_dataset, train_dataset, test_dataset
            gc.collect()
        torch.save(vit_modeling_model.state_dict(), f"model_save/vit/suffix_vit-ae_epoch_{epoch}")

    print(f"Best model is {best_model}")
