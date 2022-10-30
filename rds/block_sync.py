#!/usr/env python3
#
# This is about taking a bit stream and synchronizing decoder to locate 104-bit RDS groups, removing check bits
# and leaving pure business-layer data. So essentially producing what a software engineer would like to work with.

# "content in the offset register will not be interpreted as a burst of errors equal to or shorter than five bits
# when rotated in the polynomial shift register" WTF does that mean and why should I care?
import numpy as np
from itertools import cycle

from error_coding import build_parity_check_matrix, spec_generator_matrix, spec_generator_poly, encode, utf8_to_blocks


# one for syndrome calculation and sync, the other for actual decoding
pc_matrix = build_parity_check_matrix(26, 16, spec_generator_poly)[::-1, ::-1]
pm = build_parity_check_matrix(341, 331, spec_generator_poly)
pc_decoding_matrix = np.vstack([pm[:26, :], pm[-26:, :]])[::-1, ::-1]

# padded to length 26
offset_words = np.pad(
    np.array([
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, ],  # A
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, ],  # B
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, ],  # C
        [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, ],  # C'
        [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, ],  # D
    ]),
    ((0, 0), (16, 0)),
    'constant'
)

offset_syndromes = np.matmul(offset_words, pc_matrix) % 2


def lookup_syndrome(syndrome):
    for row, word_symbol in zip(offset_syndromes, ('A', 'B', 'C', 'Cp', 'D')):
        if np.all(row == syndrome):
            return word_symbol
    return None


offset_word_predecessors = {
    'A': {'D'},
    'B': {'A'},
    'C': {'B'},
    'Cp': {'B'},
    'D': {'C', 'Cp'}
}


# TODO: probably turn it into RadioText encoding?
# produce bit sequence of subsequent 104-bit blocks
def encode_groups(input_data):
    data_blocks = utf8_to_blocks(input_data)
    encoded_blocks = encode(data_blocks, spec_generator_matrix)
    rows = encoded_blocks.shape[0]
    for row_ind, offset_word in zip(range(rows), cycle(offset_words)):
        encoded_blocks[row_ind, :] += offset_word
    return (encoded_blocks % 2).flatten()


# TODO: add some bits, remove some bits to force Decoder to re-sync
def errors(bits):
    return bits


class Decoder:
    def __init__(self):
        self.in_sync = False
        self.current_offset = 0
        self.prev_offsets_size = 27
        self.prev_offset_words = [None] * self.prev_offsets_size
        self.errors_per_block = 0

    def decode(self, bit_arr):
        # find 2 syndromes corresponding to known offset words in correct order
        i = 0
        while i < len(bit_arr) - 26 and not self.in_sync:
            syndrome = np.matmul(bit_arr[i:i + 26], pc_matrix) % 2
            offset_word = lookup_syndrome(syndrome)
            self.prev_offset_words[self.current_offset] = offset_word

            if offset_word:
                # we check if previous offset words matched this one
                correct_predecessors = offset_word_predecessors[offset_word]
                if self.prev_offset_words[(self.current_offset - 26) % self.prev_offsets_size] in correct_predecessors:
                    self.in_sync = True
                    print('IN SYNC!')
                    break

            self.current_offset = (self.current_offset + 1) % 27
            i += 1

        # now we can start decoding the data and accumulating potential errors
        return self.current_offset


def main():
    input_data = """This is a test piece of text. It's supposed to be encoded the same as it would be by an RDS 
    transmitter and then decoded as if by an RDS receiver."""
    decoder = Decoder()

    bits = encode_groups(input_data)
    err_bits = errors(bits)
    offset = decoder.decode(err_bits)
    print(offset)


if __name__ == '__main__':
    main()
