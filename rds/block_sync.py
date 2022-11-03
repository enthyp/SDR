#!/usr/env python3
#
# This is about taking a bit stream and synchronizing decoder to locate 104-bit RDS groups, removing check bits
# and leaving pure business-layer data. So essentially producing what a software engineer would like to work with.

# "content in the offset register will not be interpreted as a burst of errors equal to or shorter than five bits
# when rotated in the polynomial shift register" WTF does that mean and why should I care?
import numpy as np
from itertools import cycle

from error_coding import (build_parity_check_matrix, decode_shortened, encode, spec_generator_matrix,
                          spec_generator_poly, blocks_to_utf8, utf8_to_blocks)


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
        # [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, ],  # C'
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


word_to_offset = {
    'A': 0, 'B': 26, 'C': 52, 'Cp': 52, 'D': 78
}

offset_to_words = {
    0: {'A'}, 26: {'B'}, 52: {'C', 'Cp'}, 78: {'D'}
}


def find_predecessors(word_symbol):
    offset = word_to_offset[word_symbol]
    return offset_to_words[(offset - 26) % 104]


def find_next_group_shift(word_symbol):
    offset = word_to_offset[word_symbol]
    return (104 - offset) % 104


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
        # synchronization
        self.in_sync = False
        self.current_offset = 0
        self.prev_offsets_size = 27
        self.prev_offset_words = [None] * self.prev_offsets_size

        # shift
        self.decoding_start = 0

        # decoding
        self.error_blocks = 0
        self.prev_bit_arr = np.array([])

    def decode(self, bit_arr):
        # steps:
        #   1. SYNC: find 2 consecutive syndromes corresponding to known offset words in correct order
        #   2. SHIFT: shift to beginning of the next group (might not be in bit_arr!)
        #   3. DECODE: decode as many groups as can fit bit_arr (accumulate decoding errors)
        #   4. SAVE: if non-decoded bits of bit_arr remain - store them for the next decoding round
        if not self.in_sync:
            # synchronize
            i = 0
            while i < len(bit_arr) - 26 and not self.in_sync:
                syndrome = np.matmul(bit_arr[i:i + 26], pc_matrix) % 2
                offset_word = lookup_syndrome(syndrome)
                self.prev_offset_words[self.current_offset] = offset_word

                if offset_word:
                    # we check if previous offset words matched this one
                    correct_predecessors = find_predecessors(offset_word)
                    if self.prev_offset_words[(self.current_offset - 26) % self.prev_offsets_size] in correct_predecessors:
                        self.in_sync = True
                        break

                self.current_offset = (self.current_offset + 1) % self.prev_offsets_size
                i += 1

            # find group start
            if self.in_sync:
                # current and (current - 26) are correct offset words - start looking from the earlier one
                shift = find_next_group_shift(self.prev_offset_words[(self.current_offset - 26) % self.prev_offsets_size])
                self.decoding_start = i - 26 + shift
            else:
                return

        # now we can start decoding the data and accumulating potential errors
        # todo: what if at this point still decoding_start >= len(bit_arr)?
        # let's add data from previous decoding call (if any)
        bit_arr = np.hstack([self.prev_bit_arr, bit_arr])
        self.prev_bit_arr = np.array([])

        if self.decoding_start >= len(bit_arr):
            self.decoding_start -= len(bit_arr)
            return

        if len(bit_arr) - self.decoding_start < 104:
            self.prev_bit_arr = bit_arr[self.decoding_start:]
            self.decoding_start = 0
            return

        decoding_rem = (len(bit_arr) - self.decoding_start) % 104  # we want to decode full groups here already
        self.prev_bit_arr = bit_arr[-decoding_rem:]
        self.decoding_start = 0

        # todo: add syndromes before decoding!
        # todo: C vs C' - distinguish by 2nd group?
        row_blocks = np.reshape(bit_arr[:len(bit_arr) - decoding_rem], (-1, 26))
        offsets = np.vstack([offset_words] * (len(row_blocks) // len(offset_words)))

        decoded = decode_shortened((row_blocks + offsets) % 2, pc_decoding_matrix, 341, 26, 331, 5)
        self.error_blocks += sum(1 for row in decoded if row is None)

        # todo: decode group data
        return decoded


def main():
    input_data = """This is a test piece of text. It's supposed to be encoded the same as it would be by an RDS 
    transmitter and then decoded as if by an RDS receiver."""
    decoder = Decoder()

    bits = encode_groups(input_data)
    err_bits = errors(bits)
    decoded_blocks = decoder.decode(err_bits)
    output = blocks_to_utf8(np.vstack(decoded_blocks))
    print(output)


if __name__ == '__main__':
    main()
