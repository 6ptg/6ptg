import json
import torch
from tqdm import tqdm

def channel0_construct(ipv6_iid):
    channel0 = []
    for i in range(64):
        temp_row = []
        for j in range(64):
            temp_row.append(ipv6_iid[j])
        channel0.append(temp_row)
    return channel0

def channel0_inverse_construct(channel):
    inverse_channel = []
    for i in range(64):
        temp_row = []
        for j in range(64):
            temp_row.append(channel[i][j])
        inverse_channel.append(temp_row)
    return inverse_channel

def channel1_construct(ipv6_iid):
    channel0 = channel0_construct(ipv6_iid)
    channel1 = []
    for i in range(64):
        temp_row = []
        for j in range(64):
            temp_row.append(~channel0[i][j] + 2)
        channel1.append(temp_row)
    return channel1

def channel1_inverse_construct(channel):
    inverse_channel = []
    for i in range(64):
        temp_row = []
        for j in range(64):
            temp_row.append(~channel[i][j] + 2)
        inverse_channel.append(temp_row)
    return inverse_channel

def channel2_construct(ipv6_iid):
    ipv6_iid_expand = []
    for i in range(2):
        for v in ipv6_iid:
            ipv6_iid_expand.append(v)
    channel2 = []
    for i in range(64):
        temp_row = []
        for j in range(i, i+64):
            temp_row.append(ipv6_iid_expand[j])
        channel2.append(temp_row)
    return channel2

def  channel2_inverse_construct(channel):
    inverse_channel = []
    for i in range(64):
        temp_row = []
        for j in range(64 - i, 64):
            temp_row.append(channel[i][j])
        for j in range(0, 64 - i):
            temp_row.append(channel[i][j])
        inverse_channel.append(temp_row)
    return inverse_channel

def channel3_construct(ipv6_iid):
    channel2 = channel2_construct(ipv6_iid)
    channel3 = []
    for i in range(64):
        temp_row = []
        for j in range(64):
            temp_row.append(~channel2[i][j] + 2)
        channel3.append(temp_row)
    return channel3

def channel3_inverse_construct(channel):
    inverse_channel = channel1_inverse_construct(channel)
    inverse_channel = channel2_inverse_construct(inverse_channel)
    return inverse_channel

def channel4_construct(ipv6_iid):
    channel2 = channel2_construct(ipv6_iid)
    channel4 = []
    for i in range(64):
        temp_row = []
        for j in range(64):
            if i == 0:
                temp_row.append(channel2[i][j])
            else:
                temp_row.append(channel2[0][j] ^ channel2[i][j])
        channel4.append(temp_row)
    return channel4

def channel4_inverse_construct(basic_row, channel):
    inverse_channel = []
    for i in range(64):
        temp_row = []
        for j in range(64):
            if i == 0:
                temp_row.append(channel[i][j])
            else:
                temp_row.append(channel[i][j] ^ basic_row[j])
        inverse_channel.append(temp_row)

    inverse_channel = channel2_inverse_construct(inverse_channel)

    return inverse_channel

def channel5_construct(ipv6_iid):
    channel4 = channel4_construct(ipv6_iid)
    channel5 = []
    for i in range(64):
        temp_row = []
        for j in range(64):
            temp_row.append(~channel4[i][j] + 2)
        channel5.append(temp_row)
    return channel5

def channel5_inverse_construct(basic_row, channel):
    inverse_channel = channel1_inverse_construct(channel)
    inverse_channel = channel4_inverse_construct(basic_row, inverse_channel)
    return inverse_channel

def ipv6_str2int(ipv6):

    output_instance = []

    for v in ipv6:
        output_instance.append(int(v))

    return output_instance

def bit_transform(ipv6_list, noise_size=1):
    prefix_list = []
    image_list = []
    for ipv6 in ipv6_list:
        prefix = ipv6_str2int(ipv6[:64])
        suffix = ipv6_str2int(ipv6[64:])
        image = []

        if noise_size == 0:
            image.append(channel0_construct(suffix))
            image.append(channel0_construct(suffix))
            image.append(channel0_construct(suffix))
        elif noise_size == 1:
            image.append(channel2_construct(suffix))
            image.append(channel3_construct(suffix))
            image.append(channel4_construct(suffix))
        else:
            image.append(channel0_construct(suffix))
            image.append(channel1_construct(suffix))
            image.append(channel2_construct(suffix))
            image.append(channel3_construct(suffix))
            image.append(channel4_construct(suffix))
            image.append(channel5_construct(suffix))
        prefix_list.append(prefix)
        image_list.append(image)
    return [prefix_list, image_list]

def single_bit_transform_suffix(ipv6, noise_size=1):
    prefix = ipv6_str2int(ipv6[:64])
    suffix = ipv6_str2int(ipv6[64:])

    image = []

    if noise_size == 0:
        image.append(channel0_construct(suffix))
        image.append(channel0_construct(suffix))
        image.append(channel0_construct(suffix))
    elif noise_size == 1:
        image.append(channel2_construct(suffix))
        image.append(channel3_construct(suffix))
        image.append(channel4_construct(suffix))
    else:
        image.append(channel0_construct(suffix))
        image.append(channel1_construct(suffix))
        image.append(channel2_construct(suffix))
        image.append(channel3_construct(suffix))
        image.append(channel4_construct(suffix))
        image.append(channel5_construct(suffix))
    return prefix, image

def single_bit_transform_prefix(ipv6, noise_size=1):
    prefix = ipv6_str2int(ipv6[:64])
    suffix = ipv6_str2int(ipv6[64:])

    image = []

    if noise_size == 0:
        image.append(channel0_construct(prefix))
        image.append(channel0_construct(prefix))
        image.append(channel0_construct(prefix))
    elif noise_size == 1:
        image.append(channel2_construct(prefix))
        image.append(channel3_construct(prefix))
        image.append(channel4_construct(prefix))
    else:
        image.append(channel0_construct(prefix))
        image.append(channel1_construct(prefix))
        image.append(channel2_construct(prefix))
        image.append(channel3_construct(prefix))
        image.append(channel4_construct(prefix))
        image.append(channel5_construct(prefix))
    return image, suffix

def single_bit_transform_mix(ipv6, noise_size=1):
    prefix = ipv6_str2int(ipv6[:64])
    suffix = ipv6_str2int(ipv6[64:])

    image_prefix = []
    image_suffix = []

    if noise_size == 0:
        image_prefix.append(channel0_construct(prefix))
        image_prefix.append(channel0_construct(prefix))
        image_prefix.append(channel0_construct(prefix))
        image_suffix.append(channel0_construct(suffix))
        image_suffix.append(channel0_construct(suffix))
        image_suffix.append(channel0_construct(suffix))
    elif noise_size == 1:
        image_prefix.append(channel2_construct(prefix))
        image_prefix.append(channel3_construct(prefix))
        image_prefix.append(channel4_construct(prefix))

        image_suffix.append(channel2_construct(suffix))
        image_suffix.append(channel3_construct(suffix))
        image_suffix.append(channel4_construct(suffix))
    else:
        image_prefix.append(channel0_construct(prefix))
        image_prefix.append(channel1_construct(prefix))
        image_prefix.append(channel2_construct(prefix))
        image_prefix.append(channel3_construct(prefix))
        image_prefix.append(channel4_construct(prefix))
        image_prefix.append(channel5_construct(prefix))

        image_suffix.append(channel0_construct(suffix))
        image_suffix.append(channel1_construct(suffix))
        image_suffix.append(channel2_construct(suffix))
        image_suffix.append(channel3_construct(suffix))
        image_suffix.append(channel4_construct(suffix))
        image_suffix.append(channel5_construct(suffix))
    return image_prefix, image_suffix

def inverse_bit_transform(image, noise_size=1):
    inverse_image = []
    if noise_size == 0:
        return image
    elif noise_size == 1:
        inverse_image.append(channel2_inverse_construct(image[0]))
        inverse_image.append(channel3_inverse_construct(image[1]))

        address_dict = {}
        for sub_image in inverse_image:
            for address in sub_image:

                address = list(map(lambda x: str(x), address))
                address = ''.join(address)
                if address not in address_dict:
                    address_dict[address] = 1
                else:
                    address_dict[address] += 1
        mx = 0
        mx_address = ''
        for address in address_dict:
            if address_dict[address] > mx:
                mx = address_dict[address]
                mx_address = address

        mx_address = list(mx_address)
        mx_address = list(map(lambda x: int(x), mx_address))

        inverse_image.append(channel4_inverse_construct(mx_address, image[2]))
    else:
        inverse_image.append(channel0_inverse_construct(image[0]))
        inverse_image.append(channel1_inverse_construct(image[1]))
        inverse_image.append(channel2_inverse_construct(image[2]))
        inverse_image.append(channel3_inverse_construct(image[3]))

        address_dict = {}
        for sub_image in inverse_image:
            for address in sub_image:

                address = list(map(lambda x: str(x), address))
                address = ''.join(address)
                if address not in address_dict:
                    address_dict[address] = 1
                else:
                    address_dict[address] += 1
        mx = 0
        mx_address = ''
        for address in address_dict:
            if address_dict[address] > mx:
                mx = address_dict[address]
                mx_address = address

        mx_address = list(mx_address)
        mx_address = list(map(lambda x: int(x), mx_address))

        inverse_image.append(channel4_inverse_construct(mx_address, image[4]))
        inverse_image.append(channel5_inverse_construct(mx_address, image[5]))
    return inverse_image



if __name__ == "__main__":
    pass



