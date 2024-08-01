from tqdm import tqdm
import ipaddress

basic_status_list = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111']

def rough_expand_suffix_construct():
    rough_suffix_set = set()
    for basic_status in basic_status_list:
        for i in range(16):
            suffix_0 = ""
            suffix_1 = ""
            for j in range(16):
                if j != i:
                    suffix_0 += '0000'
                    suffix_1 += '1111'
                else:
                    suffix_0 += basic_status
                    suffix_1 += basic_status
            rough_suffix_set.add(suffix_0)
            rough_suffix_set.add(suffix_1)
    return rough_suffix_set

def fine_expand_suffix_construct():
    fine_suffix_set = set()
    for basic_status_first in basic_status_list:
        for basic_status_second in basic_status_list:
            for i in range(15):
                for j in range(i+1, 16):
                    suffix_0 = ""
                    suffix_1 = ""
                    for k in range(16):
                        if k == i:
                            suffix_0 += basic_status_first
                            suffix_1 += basic_status_first
                        elif k == j:
                            suffix_0 += basic_status_second
                            suffix_1 += basic_status_second
                        else:
                            suffix_0 += '0000'
                            suffix_1 += '1111'
                    fine_suffix_set.add(suffix_0)
                    fine_suffix_set.add(suffix_1)
    return fine_suffix_set




if __name__ == "__main__":
    min_num = 5
    max_num = 100

    source_path = "data/init_seed_set/init_seed_set.txt"

    print("Full init active ipv6 address is reading.")

    ipv6_address_list = []
    with open(source_path, "r") as file_object:
        for content in file_object:
            if len(content) < 5:
                continue
            ipv6_address = content.replace('\n', '')
            ipv6_address_list.append(ipv6_address)

    print("The number of ipv6 address is: ", len(ipv6_address_list))

    ipv6_prefix_dict = {}
    for ipv6_address in tqdm(ipv6_address_list, desc="Prefix extracting"):
        ipv6_e = ipaddress.IPv6Address(ipv6_address).exploded
        ipv6_b = bin(int(str(ipv6_e).replace(':', ''), 16))[2:].zfill(128)
        ipv6_prefix = ipv6_b[:64]
        if ipv6_prefix not in ipv6_prefix_dict:
            ipv6_prefix_dict[ipv6_prefix] = 1
        else:
            ipv6_prefix_dict[ipv6_prefix] += 1

    rough_expand_scan_prefix_list = []
    fine_expand_scan_prefix_list = []

    cnt = 0
    for ipv6_prefix in tqdm(ipv6_prefix_dict, desc=f"Rough and fine prefix ({min_num}~{max_num} and >{max_num}) are writing"):
        if ipv6_prefix_dict[ipv6_prefix] >= min_num and ipv6_prefix_dict[ipv6_prefix] < max_num:
            rough_expand_scan_prefix_list.append(ipv6_prefix)
        elif ipv6_prefix_dict[ipv6_prefix] >= max_num:
            fine_expand_scan_prefix_list.append(ipv6_prefix)
        else:
            continue

    rough_expand_ipv6_address_list = []
    fine_expand_ipv6_address_list = []

    rough_expand_suffix_list = rough_expand_suffix_construct()
    fine_expand_suffix_list = fine_expand_suffix_construct()

    rough_expand_scan_ipv6_address_path = "data/expansion_scanning_set/rough_expand_scan.txt"
    fine_expand_scan_ipv6_address_path = "data/expansion_scanning_set/fine_expand_scan.txt"

    fo_rough_expand_scan_ipv6_address = open(rough_expand_scan_ipv6_address_path, "w", encoding='utf-8')
    fo_fine_expand_scan_ipv6_address = open(fine_expand_scan_ipv6_address_path, "w", encoding='utf-8')

    for rough_expand_prefix in tqdm(rough_expand_scan_prefix_list, desc="Rough expand scanning ipv6 address is generating"):
        for rough_expand_suffix in rough_expand_suffix_list:
            ipv6_address = rough_expand_prefix+rough_expand_suffix
            ipv6_address = ipaddress.IPv6Address(int(ipv6_address, 2))
            rough_expand_ipv6_address_list.append(str(ipv6_address))
            fo_rough_expand_scan_ipv6_address.write(str(ipv6_address)+'\n')

    for fine_expan_prefix in tqdm(fine_expand_scan_prefix_list, desc="fine expand scanning ipv6 address is generating"):
        for fine_expan_suffix in fine_expand_suffix_list:
            ipv6_address = fine_expan_prefix+fine_expan_suffix
            ipv6_address = ipaddress.IPv6Address(int(ipv6_address, 2))
            fine_expand_ipv6_address_list.append(str(ipv6_address))
            fo_fine_expand_scan_ipv6_address.write(str(ipv6_address)+'\n')

    fo_rough_expand_scan_ipv6_address.close()
    fo_fine_expand_scan_ipv6_address.close()






