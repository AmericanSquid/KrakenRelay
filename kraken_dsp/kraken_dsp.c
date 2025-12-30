// kraken_dsp.c
#include "kraken_dsp.h"
#include <math.h>

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// RBJ cookbook high-pass biquad (a0 normalized to 1)
static void biquad_highpass_rbj(float cutoff_hz, float q, float fs,
                                float* b0, float* b1, float* b2, float* a1, float* a2) {
    const float w0 = 2.0f * (float)M_PI * cutoff_hz / fs;
    const float cw = cosf(w0);
    const float sw = sinf(w0);
    const float alpha = sw / (2.0f * q);

    float _b0 = (1.0f + cw) * 0.5f;
    float _b1 = -(1.0f + cw);
    float _b2 = (1.0f + cw) * 0.5f;

    float a0  = 1.0f + alpha;
    float _a1 = -2.0f * cw;
    float _a2 = 1.0f - alpha;

    _b0 /= a0; _b1 /= a0; _b2 /= a0;
    _a1 /= a0; _a2 /= a0;

    *b0 = _b0; *b1 = _b1; *b2 = _b2;
    *a1 = _a1; *a2 = _a2;
}

// ---------- SOS filter ----------
void sos_reset(SOSFilter* f) {
    if (!f) return;
    for (int i = 0; i < KR_MAX_SECTIONS; i++) {
        f->z1[i] = 0.0f;
        f->z2[i] = 0.0f;
    }
}

void sos_init_highpass_butter(SOSFilter* f, int order, float cutoff_hz, float sample_rate) {
    if (!f) return;

    cutoff_hz = clampf(cutoff_hz, 5.0f, 0.49f * sample_rate);
    if (order != 2 && order != 4) order = 4;

    f->sections = (order == 2) ? 1 : 2;

    // Butterworth section Qs for 4th order split into two biquads
    // (common values used in practice)
    if (order == 2) {
        const float q = 0.70710678f;
        biquad_highpass_rbj(cutoff_hz, q, sample_rate,
                            &f->b0[0], &f->b1[0], &f->b2[0], &f->a1[0], &f->a2[0]);
    } else {
        const float q1 = 0.54119610f;
        const float q2 = 1.30656296f;
        biquad_highpass_rbj(cutoff_hz, q1, sample_rate,
                            &f->b0[0], &f->b1[0], &f->b2[0], &f->a1[0], &f->a2[0]);
        biquad_highpass_rbj(cutoff_hz, q2, sample_rate,
                            &f->b0[1], &f->b1[1], &f->b2[1], &f->a1[1], &f->a2[1]);
    }

    sos_reset(f);
}

void sos_process_inplace(SOSFilter* f, float* x, int n) {
    if (!f || !x || n <= 0) return;

    for (int s = 0; s < f->sections; s++) {
        float b0 = f->b0[s], b1 = f->b1[s], b2 = f->b2[s];
        float a1 = f->a1[s], a2 = f->a2[s];
        float z1 = f->z1[s], z2 = f->z2[s];

        for (int i = 0; i < n; i++) {
            float in = x[i];
            if (!isfinite(in)) in = 0.0f;

            // Direct Form II Transposed
            float y = b0 * in + z1;
            z1 = b1 * in - a1 * y + z2;
            z2 = b2 * in - a2 * y;

            x[i] = y;
        }

        f->z1[s] = z1;
        f->z2[s] = z2;
    }
}

// ----------------- Compressor ----------------
void compressor_reset(CompressorState* st) {
    st->rms_env = 0.0f;
    st->gain = 1.0f;
}

void compressor_process_inplace(
    float* x, int n,
    float threshold_db,
    float ratio,
    float attack_coeff,
    float release_coeff,
    float makeup_gain,
    CompressorState* st
) {
    float env = st->rms_env;
    float gain = st->gain;
    float threshold = powf(10.0f, threshold_db / 20.0f);

    for (int i = 0; i < n; i++) {
        float s = x[i];
        float sq = s * s;

        if (sq > env)
            env = attack_coeff * env + (1.0f - attack_coeff) * sq;
        else
            env = release_coeff * env + (1.0f - release_coeff) * sq;

        float rms = sqrtf(env + 1e-12f);

        float desired = 1.0f;
        if (rms > threshold) {
            float over = rms / threshold;
            desired = powf(over, (1.0f / ratio) - 1.0f);
        }

        if (desired < gain)
            gain = attack_coeff * gain + (1.0f - attack_coeff) * desired;
        else
            gain = release_coeff * gain + (1.0f - release_coeff) * desired;

        x[i] = s * gain * makeup_gain;
    }

    st->rms_env = env;
    st->gain = gain;
}

// ---------- Limiter ----------
void limiter_reset(LimiterState* st) {
    if (!st) return;
    st->env = 0.0f;
    st->gain = 1.0f;
}

void limiter_process_inplace(
    float* x, int n,
    float threshold,
    float att_coeff, float rel_coeff,
    LimiterState* st
) {
    if (!x || !st || n <= 0) return;

    threshold = clampf(threshold, 0.05f, 1.0f);
    att_coeff = clampf(att_coeff, 0.0f, 0.999999f);
    rel_coeff = clampf(rel_coeff, 0.0f, 0.999999f);

    float env  = st->env;
    float gain = st->gain;
    const float max_gain = 4.0f;

    for (int i = 0; i < n; i++) {
        float s = x[i];
        if (!isfinite(s)) s = 0.0f;

        float a = fabsf(s);

        // envelope follower
        if (a > env) env = att_coeff * env + (1.0f - att_coeff) * a;
        else         env = rel_coeff * env + (1.0f - rel_coeff) * a;

        float desired = 1.0f;
        if (env > threshold && env > 0.0f) desired = threshold / env;
        desired = clampf(desired, 0.0f, max_gain);

        // smooth gain (fast down, slow up)
        if (desired < gain) gain = desired + att_coeff * (gain - desired);
        else                gain = desired + rel_coeff * (gain - desired);

        float y = s * gain;

        // soft clip
        y = threshold * tanhf(y / threshold);

        x[i] = y;
    }

    st->env  = env;
    st->gain = gain;
}

// ---------- Convert ----------
void f32_to_s16(const float* x, int n, int16_t* out) {
    if (!x || !out || n <= 0) return;

    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (!isfinite(v)) v = 0.0f;
        v = clampf(v, -1.0f, 1.0f);

        float scaled = v * 32767.0f;
        if (scaled > 32767.0f) scaled = 32767.0f;
        if (scaled < -32768.0f) scaled = -32768.0f;

        out[i] = (int16_t)lrintf(scaled);
    }
}

// ---------- Chain ----------
void dspchain_init(DSPChain* ch) {
    if (!ch) return;
    ch->hpf_enabled = 0;
    ch->compressor_enabled = 0;
    ch->limiter_enabled = 0;

    ch->hpf.sections = 0;
    sos_reset(&ch->hpf);
    compressor_reset(&ch->comp);
    limiter_reset(&ch->lim);

    ch->comp_threshold_db = -18.0f;
    ch->comp_ratio = 3.0f;
    ch->comp_attack_coeff = 0.9f;
    ch->comp_release_coeff = 0.995f;
    ch->comp_makeup_gain = 1.0f;

    ch->lim_threshold = 0.85f;
    ch->lim_att_coeff = 0.90f;
    ch->lim_rel_coeff = 0.995f;
}

void dspchain_reset(DSPChain* ch) {
    if (!ch) return;
    sos_reset(&ch->hpf);
    compressor_reset(&ch->comp);
    limiter_reset(&ch->lim);
}

void dspchain_set_hpf(DSPChain* ch, int enabled, int order, float cutoff_hz, float sample_rate) {
    if (!ch) return;
    ch->hpf_enabled = enabled ? 1 : 0;
    if (ch->hpf_enabled) {
        sos_init_highpass_butter(&ch->hpf, order, cutoff_hz, sample_rate);
    }
}

void dspchain_set_compressor(
    DSPChain* ch,
    int enabled,
    float threshold_db,
    float ratio,
    float attack_coeff,
    float release_coeff,
    float makeup_gain
) {
    ch->compressor_enabled = enabled;
    ch->comp_threshold_db = threshold_db;
    ch->comp_ratio = ratio;
    ch->comp_attack_coeff = attack_coeff;
    ch->comp_release_coeff = release_coeff;
    ch->comp_makeup_gain = makeup_gain;
}

void dspchain_set_limiter(
    DSPChain* ch,
    int enabled,
    float threshold,
    float att_coeff,
    float rel_coeff
) {
    ch->limiter_enabled = enabled;
    ch->lim_threshold = threshold;
    ch->lim_att_coeff = att_coeff;
    ch->lim_rel_coeff = rel_coeff;
}

void dspchain_process_inplace(DSPChain* ch, float* x, int n) {
    if (!ch || !x || n <= 0) return;

    if (ch->hpf_enabled)
        sos_process_inplace(&ch->hpf, x, n);

    if (ch->compressor_enabled)
        compressor_process_inplace(
            x, n,
            ch->comp_threshold_db,
            ch->comp_ratio,
            ch->comp_attack_coeff,
            ch->comp_release_coeff,
            ch->comp_makeup_gain,
            &ch->comp
        );

    if (ch->limiter_enabled)
        limiter_process_inplace(
            x, n,
            ch->lim_threshold,
            ch->lim_att_coeff,
            ch->lim_rel_coeff,
            &ch->lim
        );
}
