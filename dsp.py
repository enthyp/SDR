import numpy as np


##########
# FM radio
##########
def fm_demod(iq_samples):
    # return same-length array
    return np.insert(np.angle(iq_samples[1:] * np.conj(iq_samples[:-1])), 0, 0)


####################
# Phase-locked loops
####################
def loop_filter_params(damping_factor, noise_bandwidth, sample_rate, ped_gain=0.5, nco_gain=1):
    k_p = (4 * damping_factor * noise_bandwidth) / (ped_gain * nco_gain * (damping_factor + 1 / (4 * damping_factor)) * sample_rate)
    k_i = (4 * noise_bandwidth ** 2) / (ped_gain * nco_gain * ((damping_factor + 1 / (4 * damping_factor)) * sample_rate) ** 2)
    return k_p, k_i


def pll(input_signal, sample_rate, f_0, k_p, k_i, nco_gain):
    phases = np.zeros(len(input_signal))
    integrator_acc, phase = 0, 0
    
    for i, sample in enumerate(input_signal):
        # cos here means phases will be for use in sin function
        error = np.cos(2 * np.pi * f_0 * i / sample_rate + phase) * sample
        
        integrator_acc += (k_i * error)
        phase += (nco_gain * (integrator_acc + k_p * error))

        phases[i] = phase
        
    return phases


def costas_loop(input_signal, sample_rate, f_0, k_p, k_i, nco_gain):
    phases, msg = np.zeros(len(input_signal)), np.zeros(len(input_signal))
    integrator_acc, phase = 0, 0
    
    for i, sample in enumerate(input_signal):
        sin_res = np.sin(2 * np.pi * f_0 * i / sample_rate + phase) * sample
        cos_res = np.cos(2 * np.pi * f_0 * i / sample_rate + phase) * sample
        error = sin_res * cos_res

        integrator_acc += k_i * error
        phase += nco_gain * (integrator_acc + k_p * error)

        phases[i] = phase
        msg[i] = sin_res

    return phases, msg