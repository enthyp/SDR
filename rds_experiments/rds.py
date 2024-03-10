import numpy as np
from dataclasses import dataclass
from functools import reduce
from typing import Optional
from coding import block_syndrome, correct_block, syndrome_block_type, block_type_offset_word


##############
# Demodulation
##############
def fm_demod(iq_samples):
    # return same-length array
    return np.insert(np.angle(iq_samples[1:] * np.conj(iq_samples[:-1])), 0, 0)


#########################
# Carrier synchronization
#########################
def loop_filter_params(damping_factor, noise_bandwidth, sample_rate, ped_gain=0.5, nco_gain=1):
    k_p = (4 * damping_factor * noise_bandwidth) / (ped_gain * nco_gain * (damping_factor + 1 / (4 * damping_factor)) * sample_rate)
    k_i = (4 * noise_bandwidth ** 2) / (ped_gain * nco_gain * ((damping_factor + 1 / (4 * damping_factor)) * sample_rate) ** 2)
    return k_p, k_i


def pll(input_signal, sample_rate, base_frequency, k_p, k_i, nco_gain):
    phases, errors = np.zeros(len(input_signal)), np.zeros(len(input_signal))
    integrator_acc, phase = 0, 0
    
    for i, sample in enumerate(input_signal):
        # cos here means phases will be for use in sin function
        error = np.cos(2 * np.pi * base_frequency * i / sample_rate + phase) * sample
        
        integrator_acc += (k_i * error)
        phase += (nco_gain * (integrator_acc + k_p * error))

        phases[i] = phase
        errors[i] = error
        
    return phases, errors


def costas_loop(input_signal, sample_rate, f_0, k_p, k_i, nco_gain):
    phases, errors, msg = np.zeros(len(input_signal)), np.zeros(len(input_signal)), np.zeros(len(input_signal))
    integrator_acc, phase = 0, 0
    
    for i, sample in enumerate(input_signal):
        sin_res = np.sin(2 * np.pi * f_0 * i / sample_rate + phase) * sample
        cos_res = np.cos(2 * np.pi * f_0 * i / sample_rate + phase) * sample
        error = sin_res * cos_res

        integrator_acc += k_i * error
        phase += nco_gain * (integrator_acc + k_p * error)

        phases[i] = phase
        errors[i] = error
        msg[i] = sin_res

    return phases, errors, msg


#################
# Timing recovery
#################
def mm_sync_bpsk(signal, sample_rate, symbol_rate, alpha=0.3):
    samples_per_symbol = sample_rate / symbol_rate
    
    def bpsk_symbol(sample):
        return 2 * int(np.real(sample) > 0) - 1 + 1j * 0

    deltas = []
    sample_indexes = [0]
    i = int(samples_per_symbol)
    delta_t = 0
    
    while i < len(signal):
        sample_indexes.append(i)
        deltas.append(delta_t)
        
        sample_1, sample_2 = signal[sample_indexes[-1]], signal[sample_indexes[-2]]
        symbol_1, symbol_2 = bpsk_symbol(sample_1), bpsk_symbol(sample_2)
        
        mm_delta = np.real(sample_2 * np.conj(symbol_1) - sample_1 * np.conj(symbol_2))
        delta_t += samples_per_symbol + alpha * mm_delta
        
        i += int(delta_t)
        delta_t -= int(delta_t)
    
    return np.array(sample_indexes), np.array(deltas)


#####################
# Link-layer decoding
#####################'
def bpsk_symbols_to_bits(symbols):
    return np.array([int(symbol > 0) for symbol in symbols])


def diff_decode(bits):
    return np.abs(bits[1:] - bits[:-1])


@dataclass
class Group:
    pi: int
    block_2: Optional[int] = None
    block_3: Optional[int] = None
    block_4: Optional[int] = None
    complete: bool = False


def decode_rds_groups(data_bits, enable_error_correction=False):
    # source: EN50067 3.1, Appendix C
    def block_to_int(block):
        # 16 bits block
        block = np.packbits(block)
        return (block[0] << 8) | block[1]
    
    
    def next_block_type(prev_block_tpe, prev_block_idx, next_block_idx):
        # unify C and Cp
        if prev_block_tpe == 'Cp':
            prev_block_tpe = 'C'
    
        types = ['A', 'B', 'C', 'D']
        prev_tpe_position = types.index(prev_block_tpe)
        next_tpe_position = (prev_tpe_position + (next_block_idx - prev_block_idx) // 26) % 4
        return types[next_tpe_position]

    
    def corrected_block_with_type(block, expected_block_tpe):
        if expected_block_tpe == 'C':
            # it means C or Cp actually
            corrected_block = correct_block(block, block_type_offset_word('C'))
            if corrected_block is None:
                corrected_block = correct_block(block, block_type_offset_word('Cp'))
                return corrected_block, 'Cp'
            else:
                return corrected_block, 'C'
        else:
            corrected_block = correct_block(block, block_type_offset_word(expected_block_tpe))
            return corrected_block, expected_block_tpe

    
    in_sync = False
    prev_correct_block_idx = None
    prev_correct_block_tpe = None
    
    blocks_cnt = 0
    error_blocks_cnt = 0
    groups = []
        
    for i in range(0, len(data_bits) - 25):
        syndrome = block_syndrome(data_bits[i:i + 26])
        block_tpe = syndrome_block_type(syndrome)
    
        if not in_sync:
            # for now we only use two adjacent blocks for synchronization
            if block_tpe is not None:
                if prev_correct_block_idx is not None and i - prev_correct_block_idx == 26:
                    expected_block_tpe = next_block_type(prev_correct_block_tpe, prev_correct_block_idx, i)
                    if block_tpe == expected_block_tpe:
                        in_sync = True
                        print(f'Acquired sync at bit {i}.')
    
                prev_correct_block_idx = i
                prev_correct_block_tpe = block_tpe
    
        if in_sync:
            if (i - prev_correct_block_idx) % 26 == 0:
                # condition handles case of obtaining sync in this iteration (diff == 0) and in previous iterations
                block_correct = False
                blocks_cnt += 1
    
                expected_block_tpe = next_block_type(prev_correct_block_tpe, prev_correct_block_idx, i)
    
                if block_tpe is not None:
                    # no error correction required
                    block = data_bits[i:i + 16]
                    block_correct = block_tpe == expected_block_tpe
                elif enable_error_correction:
                    # error correction can be attempted using expected offset word
                    corrected_block, block_tpe = corrected_block_with_type(data_bits[i:i + 26], expected_block_tpe)
                    if corrected_block is not None:
                        block = corrected_block
                        block_correct = True
                
                if not block_correct:
                    error_blocks_cnt += 1
                    if groups:
                        groups[-1].complete = True   
    
                # handle synchronization loss
                if blocks_cnt == 50:
                    if error_blocks_cnt > 25:
                        in_sync = False
                        block_correct = False  # don't decode if lost sync
                        print(f'Lost sync at bit {i}: {error_blocks_cnt} errors in last {blocks_cnt} blocks.')
                    blocks_cnt = error_blocks_cnt = 0
    
                if block_correct:
                    prev_correct_block_idx = i
                    prev_correct_block_tpe = block_tpe
    
                    if block_tpe == 'A':
                        groups.append(Group(pi=block_to_int(block)))
                    elif groups and not groups[-1].complete:
                        current_group = groups[-1]
                        if block_tpe == 'B':
                            current_group.block_2 = block_to_int(block)
                        elif block_tpe in ('C', 'Cp'):
                            current_group.block_3 = block_to_int(block)
                        else:
                            current_group.block_4 = block_to_int(block)
                            current_group.complete = True

    return groups


def decode_radiotext(groups):
    # source: EN50067 3.1.5.3
    def decode_group_type(block_2):
        return (block_2 >> 11) & 0x1f  # get 5 first bits
    
    
    def decode_radiotext_index(block_2):
        return block_2 & 0xf
    
    
    def decode_radiotext_ab(block_2):
        return (block_2 >> 4) & 0x01
    
    
    def block_to_str(block):
        # characters are encoded according to EN50067 Annex E. 
        # luckily at least Latin letter codes seem to match ASCII codes, so we don't have to implement the lookup table
        return chr((block >> 8) & 0xff) + chr(block & 0xff)
    
    
    radiotext = [" "] * 64
    radiotext_ab_flag = 0
    
    for i, group in enumerate(groups):
        if group.block_2 is not None:
            group_tpe = decode_group_type(group.block_2)
            if group_tpe == 4:
                # RadioText groups 0A type (4 characters in group)
                idx = decode_radiotext_index(group.block_2)
    
                if group.block_3 is not None:
                    str_3 = block_to_str(group.block_3)
                    radiotext[4 * idx] = str_3[0]
                    radiotext[4 * idx + 1] = str_3[1]
                if group.block_4 is not None:
                    str_4 = block_to_str(group.block_4)
                    radiotext[4 * idx + 2] = str_4[0]
                    radiotext[4 * idx + 3] = str_4[1]
    
                print(''.join([c for c in radiotext if c != '\r']))  # because Jupyter sucks
                
                flag = decode_radiotext_ab(group.block_2)
                if radiotext_ab_flag != flag:
                    radiotext = [" "] * 64
                    radiotext_ab_flag = flag
                    print('-' * 64)
