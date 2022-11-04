#!/usr/env python3
# 
# Error-correcting codes related stuff.
#
# Data flow:
#  encoding:
#   - input string
#   - 16-bit data blocks (RDS)
#   - 26-bit data blocks with check vectors
#   - bit stream
#   - modulated waveform
#
#  errors creep in...
#
#  decoding:
#   - modulated waveform
#   - bit stream
#   - 26-bit data blocks with check vectors (alignment required here!)
#   - 16-bit data blocks after error correction
#   - output string!
#
# Here we only go down from string to bit stream level, errors creep in, and then back up to string.
#
# ###
# ###
# ###
#
# Coding theory
#
# Linear codes
#  - linear (n, k) code = k-dim linear subspace of F^n
#  - encoding function: multiplication by GENERATOR MATRIX G (rows are basis of code subspace)
#  - decoding function: 
#     1. multiplication by PARITY CHECK MATRIX H (rows form a subspace that's orthogonal to code subspace)
#     2. result is called syndrome and equals H * z^T = H * e^T where e is error vector
#     3. for syndrome we can have various e vectors, we select one with smallest Hamming weight (has highest probability of occurring)
#  - decoding step 3 is difficult for larger codes, hence we look for codes that are easier to decode (but the same principle of lowest Hamming weight)
#  - Hamming code, e.g. (7,4) is special: syndrome easily maps to 1-bit e vector, e.g. [1, 0, 1] syndrome => error bit on position 5,
#    that's because matrix H is selected so that columns are binary number representations
#
# Cyclic codes
#  - cyclic code = linear code but every codeword can be rotated and produce another codeword
#  - unique concept for them: we interpret them as polynomials (really equivalence classes modulo x^n - 1) over F
#  - there is a generator polynomial g(x): lowest degree in code C, unique up to scalar, divides all codewords
#    (moreover, if it divides a word, it is a codeword so there's an iff)
#  - if you find a divisor of x^n - 1 of degree k, it can be used as generator polynomial of (n, k) cyclic code :)
#  - and there's also parity check polynomial h(x) = (x^n - 1) / g(x)
#  - kinda CRUCIAL: we can have a parity check matrix H2 s.t. syndrome R(x) = (H2 * z)(x) equals z(x) mod g(x)
#    (so if we subtract this syndrome from z vector we get a codeword, i.e. divisible by g(x) - question remains if it's the most likely one)
#
#  (not so interested ATM how to implement efficient encoder w/ shift registers, matrix multiplication will do)
#  
# Decoding and burst error correction in cyclic codes
#
# Logbook:
#  24.10
#   - generator polynomial proposed by RDS standard + algorithm from doesn't seem to correct any error bursts reliably :(((
#   - found out it's not a generator polynomial in 26-dimensional space, well the algorithm is probably a bit different
#   - so I'm gonna try some polynomials from the book (not shortened, maybe that would work?)
#   - ...and HELL YEAH, it works 1000/1000 times, brilliant!
#  
#  25.10
#   - tearing hear here really, it seems that I can't correct error burst of ANY length if it contains 1 on bit 25/50/75 etc.
#     i.e. youngest bit of check bits
#   - ...and I got it? I just didn't shift it by 26, by 0 to 25 so the youngest bit never moved over the right edge...
#   - so now the only question remaining is: is it possible to make it more efficient than multiplications by the full length matrix?
#  27.10
#   - yes, a bit - by considering only non-zero coefficients in length 341 polynomial, for RDS there are 52
#     (26 originally, we rotate right by 26)
import numpy as np
import random as rnd
from poly_division import divide_mod2

input_data = """It's a small step for a man... But a giant leap for the mankind."""
spec_generator_poly = [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1]

# 16 x 26 generator matrix
spec_generator_matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, ],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, ],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, ],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, ],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, ],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, ],
])

# 26 x 10 parity check matrix
spec_parity_check_matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, ],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, ],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, ],
    [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, ],
    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, ],
    [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, ],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, ],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, ],
    [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, ],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, ],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, ],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, ],
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, ],
    [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, ],
    [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, ],
    [1, 1, 0, 0, 0, 1, 1, 0, 1, 1, ],
])


def utf8_to_blocks(string, block_length=16):
    # return array of 16-bit blocks of data
    string_bytes = string.encode('utf8')

    byte_arr = np.frombuffer(string_bytes, dtype=np.uint8)
    bit_arr = np.unpackbits(byte_arr)

    # 0 bit padding
    rem = len(bit_arr) % block_length
    if rem != 0:
        bit_arr = np.pad(bit_arr, (0, block_length - rem), 'constant')
    return np.reshape(bit_arr, (-1, block_length))


def blocks_to_utf8(blocks):
    # assemble n-k-bit blocks of data back into a UTF-8 string!
    bit_arr = blocks.flatten()
    byte_arr = np.packbits(bit_arr)
    # for weird block length padding might not be full bytes
    return byte_arr.tobytes().decode('utf8', errors='ignore').strip('\x00')  # strip optional \x00 padding


# IMPORTANT NOTE:
#  in RDS standard the leftmost bit represents the highest degree polynomial coefficient, the book does the opposite
def encode(blocks, generator_matrix):
    # return binary array
    # note: for cyclic codes this can be implemented as:
    #  1) left shift input 16-bit word by 10 bits
    #  2) for this 26-degree polynomial calculate remainder modulo g(x) and set it as last 10 bits
    #     (Theorem 8.3, corollary 2)
    return np.matmul(blocks, generator_matrix) % 2


def error_burst(blocks, length):
    # TODO: this is just 1 burst
    block_len = len(blocks[0])
    bits = blocks.flatten()
    ind = rnd.randint(0, len(bits) - length)
    burst = np.zeros(len(bits))
    burst[ind:ind + length] = np.random.randint(2, size=length)
    return np.reshape((bits + burst) % 2, (-1, block_len))


# decode cyclic code (non-shortened)
def decode(blocks, parity_check_matrix, n, k, b):
    # return binary array with errors corrected and check vectors removed
    corrected_blocks = []
    for block in blocks:
        for i in range(n):
            shift_block = np.roll(block, i)
            syndrome = np.matmul(shift_block, parity_check_matrix) % 2

            # check if syndrome is now a correctable short burst
            if np.all(syndrome == 0) or (syndrome[0] == 1 and np.all(syndrome[b:] == 0)):
                extended_syndrome = np.zeros(n)
                extended_syndrome[:n - k] = syndrome
                corrected_blocks.append(np.roll((shift_block + extended_syndrome) % 2, -i))
                break
        else:
            print('Failed to find correcting burst')

    blocks_corrected = np.vstack(corrected_blocks)
    return blocks_corrected.astype(np.uint8)[:, n - k:]


# blocks: [x, n_0], parity_check_matrix: [2 * n_0, n - k (deg gen poly)]
# parity check matrix is special - it's the modulo remainder matrix but with middle rows removed (corresponding to
# polynomial coefficients that are always 0 anyway)
def decode_shortened(blocks, parity_check_matrix, n, n_0, k, b):
    # return binary array with errors corrected and check vectors removed

    padded_blocks = np.pad(blocks, ((0, 0), (0, n_0)), 'constant')
    decoded_rows = []

    for block in padded_blocks:
        for i in range(n_0 + 1):
            shift_block = np.roll(block, -i)
            syndrome = np.matmul(shift_block, parity_check_matrix) % 2

            # check if syndrome is now a correctable short burst
            if np.all(syndrome == 0) or (syndrome[-1] == 1 and np.all(syndrome[:-b] == 0)):
                extended_syndrome = np.zeros(2 * n_0)
                extended_syndrome[-(n - k):] = syndrome
                corrected_block = np.roll((shift_block + extended_syndrome) % 2, i)
                if np.all(corrected_block[-n_0:] == 0):
                    # we obtained a word from our shortened code
                    # strip zero padding and check bits
                    decoded_rows.append(corrected_block[:n_0 - (n - k)].astype(np.uint8))
                    break
        else:
            decoded_rows.append(None)  # which means error...

    return decoded_rows


# k x n
def build_generator_matrix(n, k, generator_polynomial):
    rows = []
    for i in range(n - k, n):
        row = np.zeros(n)
        row[i] = 1
        _, remainder = divide_mod2(row, generator_polynomial)
        row[:len(remainder)] = remainder  # x^i - x^i mod g(x)
        rows.append(row)
    return np.vstack(rows)


# n x (n - k)
# deg(generator) = n - k
def build_parity_check_matrix(n, k, generator_polynomial):
    rows = [np.identity(n - k)]

    for i in range(n - k, n):
        dividend = [0] * (i + 1)
        dividend[-1] = 1
        _, remainder = divide_mod2(dividend, generator_polynomial)  # x^i mod g(x)
        rows.append(remainder)

    return np.vstack(rows)


def test_correctness(encoder, decoder, k, b, input_data, trials=100):
    print(f'Test for: {input_data}')
    data_blocks = utf8_to_blocks(input_data, block_length=k)
    assert blocks_to_utf8(data_blocks) == input_data

    encoded = encoder(data_blocks)
    assert np.array_equal(decoder(encoded), data_blocks)

    for _ in range(trials):
        transmitted = error_burst(encoded, b)
        decoded = decoder(transmitted)
        output_data = blocks_to_utf8(decoded)
        assert input_data == output_data
    print('All good.')


def plain_cyclic():
    n, k = 7, 3
    generator = [1, 0, 1, 1, 1]
    b = 2  # max burst length
    gen_matrix = build_generator_matrix(n, k, generator)
    par_check_matrix = build_parity_check_matrix(n, k, generator)

    encoder = lambda data_blocks: encode(data_blocks, gen_matrix)
    decoder = lambda transmitted: decode(transmitted, par_check_matrix, n, k, b)

    test_correctness(encoder, decoder, k, b, input_data)


def rds():
    # original code is (341, 331), shortened (26, 16)
    # yeah, so we have to pretend that inputs are from 341-dimensional space with initial positions zeroed out
    # i.e. multiplied by x^325 or sth
    b = 5
    k_0 = 16
    n_0 = 26
    n = 341
    k = n - (n_0 - k_0)
    pm = build_parity_check_matrix(341, 331, spec_generator_poly)
    # select rows corresponding to x_0, x_1, ..., x_n_0-1 and x_n-n_0, ..., x_n coefficients in modulo division
    # (all others are always zero in decoding algorithm)
    # inversion - because RDS assumes the highest powers in polynomial come first
    pm_shortened = np.vstack([pm[:n_0, :], pm[-n_0:, :]])[::-1, ::-1]

    data_blocks = utf8_to_blocks(input_data)
    assert blocks_to_utf8(data_blocks) == input_data

    encoded_blocks = encode(data_blocks, spec_generator_matrix)
    decoded_blocks = decode_shortened(encoded_blocks, pm_shortened, n, n_0, k, b)
    assert np.array_equal(decoded_blocks, data_blocks)

    transmitted = error_burst(encoded_blocks, b)  # RDS should detect and correct bursts of length up to 5 bits
    decoded = decode_shortened(transmitted, pm_shortened, n, n_0, k, b)
    output_data = blocks_to_utf8(np.vstack(decoded))

    print(output_data)
    assert input_data == output_data

    encoder = lambda data_blocks: encode(data_blocks, spec_generator_matrix)
    decoder = lambda transmitted: np.vstack(decode_shortened(transmitted, pm_shortened, n, n_0, k, b))

    test_correctness(encoder, decoder, k_0, b, input_data, trials=10000)

    print(f'Hell yeah!')


if __name__ == '__main__':
    # plain_cyclic()
    rds()
