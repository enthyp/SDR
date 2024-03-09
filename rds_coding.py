import numpy as np


def divide_mod2(dividend, divisor):
    quotient = []
    remainder = dividend.copy()

    # don't assume dividend[-1] != 0
    start = len(dividend) - 1
    while dividend[start] == 0:
        start -= 1

    for i in range(start, len(divisor) - 2, -1):
        if remainder[i] == 0:
            quotient.append(0)  # must be reversed at the end
        else:
            quotient.append(1)
            for j, coefficient in enumerate(divisor[::-1]):
                remainder[i - j] = (remainder[i - j] + coefficient) % 2
                
    return quotient[::-1], remainder[:len(divisor) - 1]


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


# generator polynomial from RDS specification
rds_spec_generator_poly = [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1]


# one for syndrome calculation and sync, the other for actual decoding
parity_check_matrix = build_parity_check_matrix(26, 16, rds_spec_generator_poly)[::-1, ::-1]
pm = build_parity_check_matrix(341, 331, rds_spec_generator_poly)
decoding_matrix = np.vstack([pm[:26, :], pm[-26:, :]])[::-1, ::-1]


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

block_type_to_offset_word = dict(zip(['A', 'B', 'C', 'Cp', 'D'], offset_words))

offset_syndromes = np.matmul(offset_words, parity_check_matrix) % 2


def block_syndrome(block):
    return np.matmul(block, parity_check_matrix) % 2


def syndrome_block_type(syndrome):
    for row, block_type in zip(offset_syndromes, ('A', 'B', 'C', 'Cp', 'D')):
        if np.all(row == syndrome):
            return block_type
    return None


def block_type_offset_word(block_type):
    return block_type_to_offset_word[block_type]


def correct_block(block, offset_word):
    # pad to length 52
    padded_block = np.pad((block + offset_word) % 2, (0, 26), 'constant')

    for i in range(27):
        shift_block = np.roll(padded_block, -i)
        syndrome = np.matmul(shift_block, decoding_matrix) % 2

        # check if syndrome is now a correctable short burst
        if np.all(syndrome == 0) or (syndrome[-1] == 1 and np.all(syndrome[:-5] == 0)):
            extended_syndrome = np.zeros(52)
            extended_syndrome[-10:] = syndrome
            corrected_block = np.roll((shift_block + extended_syndrome) % 2, i)
            if np.all(corrected_block[-26:] == 0):
                # we obtained a word from our shortened code
                # strip zero padding and check bits
                return corrected_block[:16].astype(np.uint8)
    else:
        return None
