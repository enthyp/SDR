#!/usr/env python3
#
# This is about taking a bit stream and synchronizing decoder to locate 104-bit RDS groups, removing check bits
# and leaving pure business-layer data. So essentially producing what a software engineer would like to work with.

# "content in the offset register will not be interpreted as a burst of errors equal to or shorter than five bits
# when rotated in the polynomial shift register" WTF does that mean and why should I care?
import numpy as np

from error_coding import spec_generator_matrix, rds_parity_check_matrix, encode, utf8_to_blocks, blocks_to_utf8

offset_words = [
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, ],  # A
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, ],  # B
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, ],  # C
    [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, ],  # C'
    [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, ],  # D
]

# todo invert again?
# TODO this: ok, so I gotta change my convention and reverse parity check matrix, doesn't make sense to reverse
# input so many times
offset_syndromes = np.matmul(
    np.pad(np.array(offset_words), ((0, 0), (16, 0)), 'constant'),  # pad to block length 26
    rds_parity_check_matrix
) % 2


# produce bit sequence of subsequent 104-bit blocks
def encode_groups(input_data):
    data_blocks = utf8_to_blocks(input_data)
    encoded_blocks = encode(data_blocks, spec_generator_matrix)


class Decoder:
    def __init__(self):
        self.errors_per_block = 0
        self.in_sync = False
    
    def decode(bit_arr):
        # todo:
        #  
        pass


def main():
    print(offset_syndromes)


if __name__ == '__main__':
    main()
