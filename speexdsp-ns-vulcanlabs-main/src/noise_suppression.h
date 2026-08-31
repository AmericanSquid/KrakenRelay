#ifndef __NOISE_SUPPRESSION_H__
#define __NOISE_SUPPRESSION_H__

#include <string>
#include <stdint.h>

class NoiseSuppression
{
public:
    static NoiseSuppression* create(int frame_size=256, int sample_rate=16000);

    virtual std::string process(const std::string& near) = 0;

    virtual ~NoiseSuppression() {}

    virtual void set_noise_suppress(int db) = 0;
    virtual void set_denoise(int enabled) = 0;
    virtual void set_agc(int enabled) = 0;
    virtual void set_agc_level(int sr) = 0;
    virtual void set_vad(int enabled) = 0;
};


#endif // __NOISE_SUPPRESSION_H__
