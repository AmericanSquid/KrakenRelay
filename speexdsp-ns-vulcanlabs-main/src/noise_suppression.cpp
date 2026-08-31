#include <stdint.h>
#include <string.h>

#include "noise_suppression.h"
#include "speex/speex_preprocess.h"

class NoiseSuppressionImpl : public NoiseSuppression
{
public:
    NoiseSuppressionImpl(int frame_size=256, int sample_rate=16000);
    ~NoiseSuppressionImpl();

    std::string process(const std::string& near) override;

    void set_noise_suppress(int db) override;
    void set_denoise(int enabled) override;
    void set_agc(int enabled) override;
    void set_agc_level(int sr) override;
    void set_vad(int enabled) override;

private:
    SpeexPreprocessState *st;
    int16_t *e;
    int frames;
};

NoiseSuppression* NoiseSuppression::create(int frame_size, int sample_rate)
{
    return new NoiseSuppressionImpl(frame_size, sample_rate);
}

NoiseSuppressionImpl::NoiseSuppressionImpl(int frame_size, int sample_rate)
{
    st = speex_preprocess_state_init(frame_size, sample_rate);
    frames = frame_size;
    e = new int16_t[frames];

    // Safe defaults for repeater use
    int i;

    i = 1; // enable denoise
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_DENOISE, &i);

    i = -18; // moderate suppression
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_NOISE_SUPPRESS, &i);

    i = 0; // disable AGC
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_AGC, &i);

    i = sample_rate;
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_AGC_LEVEL, &i);

    i = 0; // disable VAD
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_VAD, &i);

    i = 0; // disable dereverb
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_DEREVERB, &i);
}

NoiseSuppressionImpl::~NoiseSuppressionImpl()
{
    speex_preprocess_state_destroy(st);
    delete[] e;
}

std::string NoiseSuppressionImpl::process(const std::string& near)
{
    const int16_t *y = (const int16_t *)(near.data());
    memcpy(e, y, sizeof(int16_t) * frames);

    speex_preprocess_run(st, e);

    return std::string((const char *)e, frames * sizeof(int16_t));
}

// === Controls ===

void NoiseSuppressionImpl::set_noise_suppress(int db)
{
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_NOISE_SUPPRESS, &db);
}

void NoiseSuppressionImpl::set_denoise(int enabled)
{
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_DENOISE, &enabled);
}

void NoiseSuppressionImpl::set_agc(int enabled)
{
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_AGC, &enabled);
}

void NoiseSuppressionImpl::set_agc_level(int sr)
{
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_AGC_LEVEL, &sr);
}

void NoiseSuppressionImpl::set_vad(int enabled)
{
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_VAD, &enabled);
}
