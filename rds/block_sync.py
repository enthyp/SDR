#!/usr/env python3
#
# This is about taking a bit stream and synchronizing decoder to locate 104-bit RDS groups, removing check bits
# and leaving pure business-layer data. So essentially producing what a software engineer would like to work with.

# "content in the offset register will not be interpreted as a burst of errors equal to or shorter than five bits
# when rotated in the polynomial shift register" WTF does that mean and why should I care?
import numpy as np
import random as rnd

from error_coding import (build_parity_check_matrix, encode, spec_generator_matrix,
                          spec_generator_poly, blocks_to_utf8, utf8_to_blocks)


# one for syndrome calculation and sync, the other for actual decoding
pc_matrix = build_parity_check_matrix(26, 16, spec_generator_poly)[::-1, ::-1]

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

    for row_ind in range(rows):
        # sucks
        if row_ind % 4 == 2:
            offset_word = offset_words[2] if encoded_blocks[row_ind - 1][4] == 0 else offset_words[3]
        elif row_ind % 4 == 3:
            offset_word = offset_words[4]
        else:
            offset_word = offset_words[row_ind % 4]
        encoded_blocks[row_ind, :] += offset_word
    return (encoded_blocks % 2).flatten()


# TODO: add some bits, remove some bits to force Decoder to re-sync
def errors(bits):
    # insert random additional bit
    cut = rnd.randint(0, len(bits) // 2)
    return np.hstack([bits[:cut], np.array([1]), bits[cut:]])


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
        pm = build_parity_check_matrix(341, 331, spec_generator_poly)
        self.decoding_matrix = np.vstack([pm[:26, :], pm[-26:, :]])[::-1, ::-1]

        # todo: just implement cyclic buffer?
        self.current_error_blocks = 0
        self.error_blocks_size = 5
        self.error_blocks = [False] * self.error_blocks_size
        self.prev_bit_arr = np.array([])

    def decode(self, bit_arr):
        # steps:
        #   1. SYNC: find 2 consecutive syndromes corresponding to known offset words in correct order
        #   2. SHIFT: shift to beginning of the next group (might not be in bit_arr!)
        #   3. DECODE: decode as many groups as can fit bit_arr (accumulate decoding errors)
        #   4. SAVE: if non-decoded bits of bit_arr remain - store them for the next decoding round

        if not self.in_sync:
            self._synchronize(bit_arr)
            if not self.in_sync:
                return

        # now we can start decoding the data and accumulating potential errors
        if self.decoding_start >= len(bit_arr):
            self.decoding_start -= len(bit_arr)
            return

        # let's add data from previous decoding call (if any)
        bit_arr = np.hstack([self.prev_bit_arr, bit_arr])
        self.prev_bit_arr = np.array([])

        if len(bit_arr) - self.decoding_start < 104:
            self.prev_bit_arr = bit_arr[self.decoding_start:]
            self.decoding_start = 0
            return

        decoding_rem = (len(bit_arr) - self.decoding_start) % 104  # we want to decode full groups here already
        self.prev_bit_arr = bit_arr[len(bit_arr) - decoding_rem:]
        self.decoding_start, decoding_start = 0, self.decoding_start

        groups = self.decode_groups(bit_arr[decoding_start:len(bit_arr) - decoding_rem])
        # sync might have been lost
        if not self.in_sync:
            self.prev_bit_arr = np.array([])
            # recursion might really suck here but it's simple
            groups.extend(self.decode(bit_arr[self.decoding_start + 104 * len(groups):]))

        return groups

    def _synchronize(self, bit_arr):
        i = 0
        while i < len(bit_arr) - 26 and not self.in_sync:
            syndrome = np.matmul(bit_arr[i:i + 26], pc_matrix) % 2
            offset_word = lookup_syndrome(syndrome)
            self.prev_offset_words[self.current_offset] = offset_word

            if offset_word:
                # we check if previous offset words matched this one
                prev_offset_word = self.prev_offset_words[(self.current_offset - 26) % self.prev_offsets_size]
                correct_predecessors = find_predecessors(offset_word)
                if prev_offset_word in correct_predecessors:
                    self.in_sync = True
                    print('IN SYNC')
                    break

            self.current_offset = (self.current_offset + 1) % self.prev_offsets_size
            i += 1

        # find group start
        if self.in_sync:
            # current and (current - 26) are correct offset words
            # looking from current because it should be in bit_arr
            shift = find_next_group_shift(self.prev_offset_words[self.current_offset])
            self.decoding_start = i + shift

    def decode_groups(self, bit_arr):
        # no vectorization, it's Python anyway
        blocks = np.reshape(bit_arr, (-1, 26))
        groups = []

        for group_ind in range(len(blocks) // 4):
            a_group = self._decode_block(blocks[4 * group_ind], offset_words[0])
            b_group = self._decode_block(blocks[4 * group_ind + 1], offset_words[1])

            if b_group is not None:
                # C in version A or C' in version B
                c_word = offset_words[2] if b_group[4] == 0 else offset_words[3]
                c_group = self._decode_block(blocks[4 * group_ind + 2], c_word)
            else:
                # could decode it in 2 trials
                c_group = None
            d_group = self._decode_block(blocks[4 * group_ind + 3], offset_words[4])
            groups.append([a_group, b_group, c_group, d_group])

            if sum(self.error_blocks) / self.error_blocks_size > 0.5:
                print('LOST SYNC')
                self.in_sync = False
                self.error_blocks = [False] * self.error_blocks_size
                break

        return groups

    def _decode_block(self, block, offset_word):
        padded_block = np.pad((block + offset_word) % 2, (0, 26), 'constant')

        for i in range(27):
            shift_block = np.roll(padded_block, -i)
            syndrome = np.matmul(shift_block, self.decoding_matrix) % 2

            # check if syndrome is now a correctable short burst
            if np.all(syndrome == 0) or (syndrome[-1] == 1 and np.all(syndrome[:-5] == 0)):
                extended_syndrome = np.zeros(52)
                extended_syndrome[-10:] = syndrome
                corrected_block = np.roll((shift_block + extended_syndrome) % 2, i)
                if np.all(corrected_block[-26:] == 0):
                    # we obtained a word from our shortened code
                    # strip zero padding and check bits
                    self._update_errors(False)
                    return corrected_block[:16].astype(np.uint8)
        else:
            self._update_errors(True)
            return None  # which means error...

    def _update_errors(self, is_error):
        self.error_blocks[self.current_error_blocks] = is_error
        self.current_error_blocks = (self.current_error_blocks + 1) % self.error_blocks_size


def test1(input_data):
    decoder = Decoder()

    # kinda like a stream
    bits = encode_groups(input_data + input_data)
    err_bits = errors(bits)
    decoded_groups = decoder.decode(err_bits)

    # test if single call works
    output = blocks_to_utf8(np.vstack([block for group in decoded_groups for block in group if block is not None]))
    print(output)


def test2(input_data):
    bits = encode_groups(input_data + input_data)
    decoder = Decoder()

    # test if decoding message split into multiple separate calls works
    cut = len(bits) // 3
    assert cut % 104 != 0  # make sure it's not nice even splits

    g1 = decoder.decode(bits[:cut])
    g2 = decoder.decode(bits[cut:2 * cut])
    g3 = decoder.decode(bits[2 * cut:])

    output2 = blocks_to_utf8(np.vstack([block for group in g1 + g2 + g3 for block in group]))
    assert output2 in (input_data + input_data)


def main():
    input_data = """This is a test piece of text. It's supposed to be encoded the same as it would be by an RDS 
    transmitter and then decoded as if by an RDS receiver."""

    test1(input_data)
    test2(input_data)


if __name__ == '__main__':
    main()
