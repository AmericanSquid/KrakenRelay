from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef("""
#define KR_MAX_SECTIONS 8

typedef struct {
    int sections;
    float b0[KR_MAX_SECTIONS];
    float b1[KR_MAX_SECTIONS];
    float b2[KR_MAX_SECTIONS];
    float a1[KR_MAX_SECTIONS];
    float a2[KR_MAX_SECTIONS];
    float z1[KR_MAX_SECTIONS];
    float z2[KR_MAX_SECTIONS];
} SOSFilter;

typedef struct {
    float rms_env;
    float gain;
} CompressorState;

typedef struct {
    float env;
    float gain;
} LimiterState;

typedef struct {
    int hpf_enabled;
    SOSFilter hpf;

    int compressor_enabled;
    CompressorState comp;

    int limiter_enabled;
    LimiterState lim;

    float comp_threshold_db;
    float comp_ratio;
    float comp_attack_coeff;
    float comp_release_coeff;
    float comp_makeup_gain;

    float lim_threshold;
    float lim_att_coeff;
    float lim_rel_coeff;
} DSPChain;

void dspchain_init(DSPChain* ch);
void dspchain_reset(DSPChain* ch);

void dspchain_set_hpf(DSPChain* ch, int enabled, int order, float cutoff_hz, float sample_rate);
void dspchain_set_compressor(DSPChain* ch, int enabled, float threshold_db, float ratio,
                             float attack_coeff, float release_coeff, float makeup_gain);
void dspchain_set_limiter(DSPChain* ch, int enabled, float threshold, float att_coeff, float rel_coeff);

void dspchain_process_inplace(DSPChain* ch, float* x, int n);
""")

ffibuilder.set_source(
    "_kraken_dsp",
    '#include "kraken_dsp.h"',
    sources=["kraken_dsp.c"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
