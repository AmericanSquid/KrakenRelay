# KrakenRelay - Open Source Repeater Controller

**KrakenRelay** is an open-source repeater controller application built with Python and PyQt5. It allows users to manage radio repeaters with advanced audio processing, squelch control, CTCSS tone detection, Morse code ID transmission, and more.

This project is inspired by [**SvxLink**](https://github.com/sm0svx/svxlink) (by **SM0SVX**) and provides a simplified, user-friendly interface for basic repeater or crossband operations. While SvxLink excels with advanced features like EchoLink, this project focuses on offering a more streamlined solution, removing unnecessary options for these specific applications. It’s designed to avoid the need for command-line configuration, making setup easier and more accessible. This project is not intended to replace SvxLink but rather to provide an alternative for users seeking a simpler, more focused implementation.

## Features

- 📡 **Repeater Controller:** Routes audio from one radio on the input frequency and plays it back in real-time through the output frequency radio.
- 🔊 **CTCSS Tone Detection:** Detect and filter CTCSS tones—this is how the PTT is controlled in the software. It does not start replaying through the output until the correct PL is detected.
- 🎙️ **Morse Code ID & Courtesy Tone:** Automatically transmits CW ID at set intervals and includes a preconfigured courtesy tone.
- 🎛️ **Noise Reduction:** Includes a high pass filter, noise gate and "squelch" to filter our interference and digital artifacts.
- 🖥️ **Graphical Interface:** PyQt5-based UI for easy configuration and monitoring.
- 🔄 **Cross-Platform:** Works on Windows, macOS, and Linux.

---

## Installation

### **Prerequisites**

Ensure you have the following installed:

- Python 3.8+
- Pip (Python Package Manager)
- PortAudio (for PyAudio support)

### **Clone the Repository**

```sh
git clone https://github.com/yourusername/KrakenRelay.git
cd KrakenRelay
```

### **Install Dependencies**

```sh
pip install -r requirements.txt
```

### **Running the Application**

```sh
python main.py
```

---

## Building a Standalone Executable

You can package the application into a standalone executable using PyInstaller:

```sh
pyinstaller --onefile --windowed KrakenRelay.spec
```

For Windows, this will create `dist/KrakenRelay.exe`.

For macOS/Linux, an equivalent binary will be generated in the `dist/` folder.

---

## Configuration

The application uses `config.yaml` for settings. Below is a breakdown of the available configurations:

| Configuration               | Description                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------ |
| **chunk\_size**             | Size of the audio buffer for processing. Default: `1024`                             |
| **highpass\_cutoff**        | Frequency cutoff for the high-pass filter (Hz). Default: `300`                       |
| **highpass\_enabled**       | Enables/disables the high-pass filter. Default: `false`                              |
| **input\_gain**             | Adjusts input gain (dB). Default: `0`                                                |
| **noise\_gate\_enabled**    | Enables/disables noise gate filtering. Default: `false`                              |
| **noise\_gate\_threshold**  | Sets the noise gate threshold level. Default: `500`                                  |
| **output\_gain**            | Adjusts output gain (dB). Default: `0`                                               |
| **sample\_rate**            | Audio sampling rate (Hz). Default: `48000`                                           |
| **squelch\_threshold**      | dB level threshold required for audio to pass through. Default: `-40`                |
| **cw\_enabled**             | Enables/disables CW ID. Default: `true`                                              |
| **interval\_minutes**       | Interval for transmitting CW ID (minutes). Default: `10`                             |
| **anti\_kerchunk\_time**    | Time delay before allowing transmission (seconds). Default: `1.0`                    |
| **callsign**                | The callsign transmitted in the CW ID.                                               |
| **carrier\_delay**          | Delay before transmitting carrier signal (seconds). Default: `0.25`                  |
| **courtesy\_tone\_enabled** | Enables/disables the courtesy tone. Default: `true`                                  |
| **cw\_pitch**               | Frequency of CW ID tone (Hz). Default: `800`                                         |
| **cw\_wpm**                 | Speed of CW ID in words per minute. Default: `20`                                    |
| **pl\_threshold**           | Threshold for CTCSS tone detection. Default: `0.1`                                   |
| **pl\_tone\_freq**          | CTCSS tone frequency (Hz). Default: `141.3`                                          |
| **tail\_time**              | Time before transmission ends after the last detected signal (seconds). Default: `2` |

---

## Usage Guide

### General Instructions

1. **Connect Two Radios to the Soundcard**  
   - Use audio cables to connect any two radios that support such connections.

2. **Input Frequency Radio Setup**  
   - Connect the radio on the input frequency so that its audio output goes to an audio input on the computer.  
   - If possible, completely disable the transmit functionality on this radio.

3. **Output Frequency Radio Setup**  
   - Connect the radio on the output frequency so that the PC’s audio output goes to a mic input on the radio.  
   - Set this radio to VOX mode.  
   - Use a fairly sensitive VOX setting—this setting may vary by radio, so you'll need to test and adjust for your specific setup.

4. **Optimize Audio Levels**  
   - Adjust audio levels in your system’s mixer and on the input frequency radio to achieve optimal audio quality.  
   - While the console will log alerts for distortion, fine-tuning by ear is still required for the best results.

### UV-5R Specific Instructions

1. **VOX Setting on the UV-5R**  
   - A recommended starting point for the UV-5R is a VOX value of 2. (Remember, this might vary, so further adjustment may be needed.)

2. **Disabling RX on the UV-5R**  
   - If possible, cut off the receive functionality on the output radio.  
   - On the UV-5R, this can be achieved using CHIRP by setting all squelch values to 123.  
   - This step prevents accidental transmissions and interference that could potentially jam the repeater.

### Software Setup

1. **Set Input and Output Devices** in the UI. Selecting "Default" will use the currently active devices, which is the easiest and most reliable method.
2. **Adjust Noise Reduction Settings** (Squelch, High-pass filter, Noise Gate).
3. **Configure Repeater Settings:**
   - Enter your callsign for the CW IDer if you wish to enable CW ID.
   - Choose the appropriate PL tone for your repeater system.
   - Ensure repeater users configure their radios with the correct transmit PL tone.
   - Matching RX and TX CTCSS tones on both radios helps keep unwanted transmissions out.
4. **Click "Start" to begin operating the repeater.**

The settings autosave after entry.

---

## Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Added a new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

---

## License

This project is licensed under the GNU General Public License (v3). See `LICENSE` for details.

---

## Contact

📧 **Email:** [matthew@americansquid.com](mailto:matthew@americansquid.com)  
🐙 **GitHub:** [American Squid](https://github.com/americansquid)
