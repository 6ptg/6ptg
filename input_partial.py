from tqdm import tqdm
import ipaddress
import multiprocessing
from model.tools import get_file_path

def ip_expanding(output_path, ipv6_address_list):
    output_file = open(output_path, "w", encoding='utf-8')
    for ipv6_address in ipv6_address_list:
        ipv6_e = ipaddress.IPv6Address(ipv6_address).exploded
        ipv6_b = bin(int(str(ipv6_e).replace(':', ''), 16))[2:].zfill(128)
        output_file.write(ipv6_b)
        output_file.write('\n')
    output_file.close()

if __name__ == "__main__":
    n_cpus = 32
    max_ip_num = 128000
    threshold = 100

    ipv6_address_list_t = get_file_path("data/final_seed_set")
    ipv6_address_list = []
    for file in ipv6_address_list_t:
        tf = file.split('/')[-1][:-4]
        ipv6_address_list.append(file)

    ip_list = []
    for file in tqdm(ipv6_address_list, desc="file traveling"):
        with open(file, "r") as file_object:
            for content in file_object:
                if len(content) < 5:
                    continue
                ip_list.append(content.replace('\n', ''))

    ipv6_address_list = ip_list

    print("The number of ipv6 address is: ", len(ipv6_address_list))

    pool = multiprocessing.Pool(n_cpus)
    mp_res = []

    ipv6_address_list_part = []
    for i in range(0, len(ipv6_address_list), max_ip_num):
        ipv6_address_list_part.append(ipv6_address_list[i:min(len(ipv6_address_list), i + max_ip_num)])

    pbar = tqdm(total=len(ipv6_address_list_part))
    pbar.set_description("IP expanding and partial writing")
    update = lambda *args: pbar.update()

    index = 0
    for ipv6_list_part in ipv6_address_list_part:
        output_path = "data/input_partial"
        mp_res.append(pool.apply_async(ip_expanding, (output_path + f"/input_partial_{index}_{min(len(ipv6_address_list), index + max_ip_num)}.txt", ipv6_list_part,), callback=update))
        index += max_ip_num

    pool.close()
    pool.join()
    pbar.close()

    for res in mp_res:
        res.get()