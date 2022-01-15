# Steps
1. Get some FM radio data with RDS available:
  - find with gqrx
  - save with `rtl_sdr -f 92.5M -s 2.4M -n 24000000 filename` - used 92.5MHz (Radio Wanda Złote Przeboje), 1.8MHz, 54000000 (30s duration)
    - note: samples are saved as pairs (I/Q) of 8-bit integers (**can infer from filesize - it's 2x num_samples**)
2. FM demodulation with `rtl_fm`:
  - command that works (without wideband-FM mode it won't work! pure static noise): `rtl_fm -M wbfm  -s 1800000 -f 92500000  | play -t raw -r 32k -es -b 16 -c 1 -V1 -`
  - just fine too: `rtl_fm -M fm -s 170k -r 32k -f 92500000 | play -t raw -r 32k -es -b 16 -c 1 -V1 -`
3. RDS with Redsea:
  - command that works (171k is necessary!): `rtl_fm -M fm -s 171k -f 92.5M  | redsea`

# Notes
 - complex signal (e.g. I/Q representation) sampled at Fs can represent frequency range [-Fs/2, Fs/2] - Nyquist holds, because
   each complex sample contains 2 real samples (~doubled sampling rate)

# Questions
 - `rtl_fm`:
   - why -s 170k is louder (and cleaner?) than -s 1800k?
   - possible answer: RTL SDR is set to sample around 1e6/s and downsample(!) to -s - does it handle such large values?
     - this is effectively bandwidth we take to demodulation too? (170k is reasonable, GQRX uses 160k by default)
   - does it tune to other (close) frequency to avoid DC spike interference?

# Log

### 09.01.22
**Figuring out how `rtl_fm` works**
 - settings:
   - demod.{rate_in, rate_out} = 171k, demod.rate_out2 = output.rate = 32k
   - demod.downsample = (1_000_000 / 170_000) + 1 (6 here) - in `controller_thread_fn`
   - demod.output_scale = 1 (for mode: FM) - in `controller_thread_fn`
   - `controller_thread_fn` - periodically changes dongle's freq and rate - WHY?
     - `dongle_state.rate` - `demod.downsample * demod.rate_in` - approx. 1_000_000...
     - `dongle_state.freq` - `frequency + dongle_state.rate / 4` (`controller_state.edge` is 0 without `-E edge`)
 - `full_demod` - the meat:
   - `low_pass`:
     - interesting: it groups RTL I/Q samples into packs of `demod.downsample` length and adds them - 6x shorter output
     - interesting because it's not convolution (not really FIR box filter)
       - **update**: it is convolution followed by downsampling (pretty ingenious)
     - input and result in `demod_state.lowpassed`
   - `fm_demod`:
     - moves (I/Q) sample by sample, computes phase difference between current and prev
       - atan2 + multiplication by complex conjugate
     - phase diff in [-pi, pi] -> divide by pi, multiply by 2^14 (WHY 14 and not 15?), cast to int16
   - `low_pass_real` (because `rate_out2`):
     - like `low_pass`, groups into packs of approx. `demod.rate_out / demod.rate_out2` samples (5 here) and AVERAGES them
       - still not full convolution (followed by decimation as above)
     - casts to int16
     
### 13.01.22
**Successful demodulation of Mambo no. 5**
 - implemented 2 algos - `rtl_fm` (works) and Lab6 (doesn't work)
 - turns out SNR of ~20dB is required for decent quality (antenna higher, closer to window), previous sample was about ~10dB and barely intelligible
 - **but** `rtl_fm -M fm -s 170k -r 32k -f 92500000 | play -t raw -r 32k -es -b 16 -c 1 -V1 -` does it with antenna in default location too (so I GUESS around 10 dB as well) - but **how?**
   - settings of `rtl_sdr` - e.g. tuning to different frequency and then shifting?
   - de-emphasis filter?
   - these rectangular simple decimator filters are actually better than `scipy.signal.decimate`???
   - `rtl_fm` uses 170 kHz instead of my 240 kHz for filter bandwidth (note: with 1.8MHz I have used 300 kHz and it yielded even worse results - maybe that's the thing??)
# SDR
