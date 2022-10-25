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
#
import numpy as np
import random as rnd
from poly_division import divide_mod2


input_data = """It's a small step for a man... But a giant leap for the mankind."""
spec_generator_poly = [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1]

# 16 x 26 generator matrix
spec_generator_matrix = np.array([
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,], 
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,1,1,1,],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,1,1,1,1,],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,1,1,],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,1,0,0,1,],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,0,1,1,1,],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,1,1,1,1,1,1,],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,1,1,],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,1,1,1,0,1,],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,1,1,0,0,1,0,],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,1,1,0,0,1,],
])

# 26 x 10 parity check matrix
spec_parity_check_matrix = np.array([ 
    [1,0,0,0,0,0,0,0,0,0,], 
    [0,1,0,0,0,0,0,0,0,0,],
    [0,0,1,0,0,0,0,0,0,0,],
    [0,0,0,1,0,0,0,0,0,0,],
    [0,0,0,0,1,0,0,0,0,0,],
    [0,0,0,0,0,1,0,0,0,0,],
    [0,0,0,0,0,0,1,0,0,0,],
    [0,0,0,0,0,0,0,1,0,0,],
    [0,0,0,0,0,0,0,0,1,0,],
    [0,0,0,0,0,0,0,0,0,1,],
    [1,0,1,1,0,1,1,1,0,0,],
    [0,1,0,1,1,0,1,1,1,0,],
    [0,0,1,0,1,1,0,1,1,1,],
    [1,0,1,0,0,0,0,1,1,1,],
    [1,1,1,0,0,1,1,1,1,1,],
    [1,1,0,0,0,1,0,0,1,1,],
    [1,1,0,1,0,1,0,1,0,1,],
    [1,1,0,1,1,1,0,1,1,0,],
    [0,1,1,0,1,1,1,0,1,1,],
    [1,0,0,0,0,0,0,0,0,1,],
    [1,1,1,1,0,1,1,1,0,0,],
    [0,1,1,1,1,0,1,1,1,0,],
    [0,0,1,1,1,1,0,1,1,1,],
    [1,0,1,0,1,0,0,1,1,1,],
    [1,1,1,0,0,0,1,1,1,1,],
    [1,1,0,0,0,1,1,0,1,1,],
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


# IMPORTANT NOTE:
#  in RDS standard the leftmost bit represents highest degree polynomial coefficient, the book does the opposite
def encode(blocks, generator_matrix):
    # return binary array
    # note: for cyclic codes this can be implemented as:
    #  1) left shift input 16-bit word by 10 bits
    #  2) for this 26-degree polynomial calculate remainder modulo g(x) and set it as last 10 bits (Theorem 8.3, corrolary 2)
    encoded_blocks = np.matmul(blocks, generator_matrix)
    return encoded_blocks.flatten() % 2


def error_burst(bits, length):
    ind = rnd.randint(0, len(bits) - length)
    burst = np.zeros(len(bits))
    x = np.random.randint(2, size=length)
    ind = 25
    burst[ind:ind + length] = np.ones(length)
    return (bits + burst) % 2


def decode(bits, parity_check_matrix, b):
    # return binary array with errors corrected and check vectors removed
    n, check_size = parity_check_matrix.shape
    k = n - check_size

    bit_blocks = np.reshape(bits, (-1, n))
    
    corrected_rows = []
    for row in bit_blocks:
        for i in range(n):
            shift_row = np.roll(row, i)
            syndrome = np.matmul(shift_row, parity_check_matrix) % 2

            if np.all(syndrome == 0) or (syndrome[0] == 1 and np.all(syndrome[b:] == 0)):  # check if syndrome is now a correctable short burst
                extended_syndrome = np.zeros(n)
                extended_syndrome[:check_size] = syndrome
                corrected_rows.append(np.roll((shift_row + extended_syndrome) % 2, -i))
                break
        else:
            print('Failed to find correcting burst')

    bit_blocks_corrected = np.vstack(corrected_rows)
    return bit_blocks_corrected.astype(np.uint8)[:, check_size:]


# for input from RDS crappy reversed generator matrix + handle shortened code in a dumb way to test RDS
def decode_reverse(bits, parity_check_matrix, n_0, b):
    # return binary array with errors corrected and check vectors removed
    n, check_size = parity_check_matrix.shape
    k = n - check_size

    bit_blocks = np.reshape(bits, (-1, n_0))
    
    corrected_rows = []
    for row in bit_blocks:
        row = np.pad(row[::-1], (n - n_0, 0), 'constant')
        for i in range(n_0 + 1):
            shift_row = np.roll(row, i)
            syndrome = np.matmul(shift_row, parity_check_matrix) % 2

            if np.all(syndrome == 0) or (syndrome[0] == 1 and np.all(syndrome[b:] == 0)):  # check if syndrome is now a correctable short burst
                extended_syndrome = np.zeros(n)
                extended_syndrome[:check_size] = syndrome
                corrected_row = np.roll((shift_row + extended_syndrome) % 2, -i)
                if np.all(corrected_row[:n_0] == 0):
                    # we obtained a word from our shortened code
                    corrected_rows.append(corrected_row[-n_0:])
                    break
        else:
            print('Failed to find correcting burst')

    bit_blocks_corrected = np.vstack(corrected_rows)[:, ::-1]
    return bit_blocks_corrected.astype(np.uint8)[:, :-check_size]


def blocks_to_utf8(blocks):
    # assemble n-k-bit blocks of data back into a UTF-8 string!
    bit_arr = blocks.flatten()
    byte_arr = np.packbits(bit_arr)
    return byte_arr.tobytes().decode('utf8').strip('\x00')   # strip optional \x00 padding


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
        _, remainder = divide_mod2(dividend, generator_polynomial)
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
    decoder = lambda transmitted: decode(transmitted, par_check_matrix, b)

    test_correctness(encoder, decoder, k, b, input_data)


def rds():
    # original code is (341, 331), shortened (26, 16)
    # yeah, so we have to pretend that inputs are from 341-dimensional space with initial posiitons zeroed out
    # i.e. multiplied by x^325 or sth
    b = 5
    k = 16
    n_0 = 26
    pm = build_parity_check_matrix(341, 331, spec_generator_poly)
    
    # pm_shortened = build_parity_check_matrix(26, 16, spec_generator_poly)
    # pm_rolled = np.roll(pm, 331, axis=0)  # unit matrix to the bottom
    # pm_shortened = np.vstack([pm[:26, :], pm[315:, :]])
     
    data_blocks = utf8_to_blocks(input_data)
    assert blocks_to_utf8(data_blocks) == input_data
    
    encoded = encode(data_blocks, spec_generator_matrix)
    assert np.array_equal(decode_reverse(encoded, pm, n_0, b), data_blocks)

    transmitted = error_burst(encoded, b)  # RDS should detect and correct bursts of length up to 5 bits
    decoded = decode_reverse(transmitted, pm, n_0, b) 
    output_data = blocks_to_utf8(decoded) 

    print(output_data)
    assert input_data == output_data

    encoder = lambda data_blocks: encode(data_blocks, spec_generator_matrix)
    decoder = lambda transmitted: decode_reverse(transmitted, pm, n_0, b)

    test_correctness(encoder, decoder, k, b, input_data, trials=1000)

    print(f'Hell yeah!')


if __name__ == '__main__':
    # plain_cyclic()
    rds()

