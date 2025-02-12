import numpy as np

class MorseCode:
    MORSE_CODE = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
        'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
        'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
        'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
        'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
        'Z': '--..', '0': '-----', '1': '.----', '2': '..---', '3': '...--',
        '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
        '9': '----.'
    }

    def __init__(self, wpm=20, frequency=800, sample_rate=8000):
        self.dot_length = int(1.2 / wpm * sample_rate)
        self.dash_length = self.dot_length * 3
        self.frequency = frequency
        self.sample_rate = sample_rate

    def generate(self, text):
        t = np.arange(self.dot_length) / self.sample_rate
        dot = np.sin(2 * np.pi * self.frequency * t)
        dash = np.sin(2 * np.pi * self.frequency * np.arange(self.dash_length) / self.sample_rate)
        
        output = np.array([])
        space = np.zeros(self.dot_length)
        
        for char in text.upper():
            if char in self.MORSE_CODE:
                for symbol in self.MORSE_CODE[char]:
                    output = np.append(output, dot if symbol == '.' else dash)
                    output = np.append(output, space)
                output = np.append(output, space * 2)
                
        return (output * 32767).astype(np.int16)
