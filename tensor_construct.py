from model import dataset
import multiprocessing
from tqdm import tqdm
from model import bit_transform
import torch
from model.tools import get_file_path
import os

def ip_to_image(input_file, output_path, noise_size, num_workers):

    ipv6_address_list = []

    length = 0
    with open(input_file, 'r') as file_object:
        for content in file_object:
            length += 1

    pool = multiprocessing.Pool(num_workers)
    mp_res = []

    pbar = tqdm(total=length)
    pbar.set_description("Dataset is reading")
    with open(input_file, 'r') as file_object:
        for content in file_object:
            pbar.update()
            if len(content) < 5:
                continue
            ipv6_address_list.append(content.replace('\n', ''))
    pbar.close()

    pbar = tqdm(total=length)
    pbar.set_description("Dataset is constructing")
    update = lambda *args: pbar.update()

    for ipv6_address in ipv6_address_list:
        mp_res.append(
            pool.apply_async(bit_transform.single_bit_transform_mix, (ipv6_address, noise_size,), callback=update))

    pool.close()
    pool.join()
    pbar.close()

    prefix_list = []
    suffix_list = []

    for res in mp_res:
        prefix, suffix = res.get()
        prefix_list.append(prefix)
        suffix_list.append(suffix)

    prefix_image_list = torch.tensor(prefix_list, dtype=torch.float32)
    suffix_image_list = torch.tensor(suffix_list, dtype=torch.float32)

    file_name_prefix = input_file.split('/')[-1][:-4]
    os.makedirs(f"{output_path}/{file_name_prefix}", exist_ok=True)

    torch.save(prefix_image_list, f"{output_path}/{file_name_prefix}/prefix.pt")
    torch.save(suffix_image_list, f"{output_path}/{file_name_prefix}/suffix.pt")


def tensor_dataset_generation(source_file_list_path, noise_size, num_workers):

    target_file_list_path = "data/model_input"
    os.makedirs(target_file_list_path, exist_ok=True)
    file_path_list = get_file_path(source_file_list_path)
    for input_file_path in file_path_list:
        ip_to_image(input_file_path, target_file_list_path, noise_size, num_workers)

if __name__ == "__main__":
    noise_size = 1
    num_workers = 32
    dataset = "data/input_partial"
    tensor_dataset_generation(dataset, noise_size, num_workers)




