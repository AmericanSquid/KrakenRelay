import logging

class TonePlayer:
    def __init__(self, config, send_pcm_callable, tx_state_callable, tone_generator):
        self.config = config
        self.is_transmitting = tx_state_callable
        self.send_pcm = send_pcm_callable
        self.generator = tone_generator
        self.courtesy_tone = self.generator.generate_courtesy_tone()
        self.tot_tone = self.generator.generate_tot_tone()

    def play_courtesy_tone(self):
        repeater_cfg = self.config.config['repeater']

        if repeater_cfg['courtesy_tone_enabled']:
            self.send_pcm(self.courtesy_tone)
            logging.info("Played courtesy tone")

    def play_tot_tone(self):
        if self.is_transmitting():
            self.send_pcm(self.tot_tone)
            logging.info("Played TOT warning tone")
