// kraken_dsp.h
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KR_MAX_SECTIONS 8

// ---- SOS-style filter (biquad cascade) ----
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

void sos_reset(SOSFilter* f);
void sos_init_highpass_butter(SOSFilter* f, int order, float cutoff_hz, float sample_rate);
void sos_process_inplace(SOSFilter* f, float* x, int n);

// ---- Compressor ----
typedef struct {
    float rms_env;
    float gain;
} CompressorState;

void compressor_reset(CompressorState* st);

void compressor_process_inplace(
    float* x, int n,
    float threshold_db,
    float ratio,
    float attack_coeff,
    float release_coeff,
    float makeup_gain,
    CompressorState* st
);

// ---- Limiter ----
typedef struct {
    float env;
    float gain;
} LimiterState;

void limiter_reset(LimiterState* st);
void limiter_process_inplace(
    float* x, int n,
    float threshold,
    float att_coeff, float rel_coeff,
    LimiterState* st
);

// Convert normalized float32 [-1..1] -> int16
void f32_to_s16(const float* x, int n, int16_t* out);

// ---- DSP Chain (HPF -> Compressor -> Limiter) ----
typedef struct {
    int hpf_enabled;
    SOSFilter hpf;

    int compressor_enabled;
    CompressorState comp;

    int limiter_enabled;
    LimiterState lim;

    // compressor params
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
void dspchain_set_compressor(DSPChain* ch, int enabled, float threshold_db, float ratio, float attack_coeff, float release_coeff, float makeup_gain);
void dspchain_set_limiter(DSPChain* ch, int enabled, float threshold, float att_coeff, float rel_coeff);

// In-place processing on normalized float32 buffer
void dspchain_process_inplace(DSPChain* ch, float* x, int n);

#ifdef __cplusplus
}
#endif
