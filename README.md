# KrakenRelay - Open Source Repeater Controller

KrakenRelay is an open-source, Python-based repeater controller built with PyQt5 and designed by K3AYV. It is currently in use on the WB3DZO repeater system (147.030+ / 448.325−) in Baltimore, MD.

This controller is designed for **simplicity, reliability, and versatility** — ideal for:

- 🛰️ **Small personal repeaters**
- 🏕️ **Temporary field deployments**
- 🏙️ **Full-scale repeater systems with remote sites**

Whether you're building a basic crossband system or managing a larger linked repeater network, KrakenRelay offers a streamlined and customizable alternative to larger, more complex systems.

---

## Features

- 📡 **Repeater Control**  
  Routes audio from one radio to another, enabling traditional repeater functionality or crossbanding.

- 🎙️ **CW ID & Courtesy Tone**  
  Sends Morse code ID at configurable intervals with a customizable courtesy tone.

- 🎛️ **Basic DSP Processing**  
  Includes a high-pass filter and software-based squelch.

- 🖥️ **Graphical User Interface**  
  PyQt5-based UI makes configuration fast and intuitive.

- 🔄 **Cross-Platform**  
  Works on Linux, Windows, and macOS (Linux recommended for stable deployment).

---

## Installation

### Prerequisites

Ensure the following are installed on your system:

- Python 3.8+
- Pip (Python Package Manager)
- PortAudio (required for PyAudio)

### Clone the Repository

```bash
git clone https://github.com/yourusername/KrakenRelay.git
cd KrakenRelay
```

### **Install Dependencies**

```sh
sudo apt update
sudo apt install portaudio19-dev
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

## Usage Guide

### Connect Radios

1) Attach an input and output radio to your computer via sound card.

  - Your input radio should feed audio into the PC.

  - Your output radio should receive audio from the PC (line out → mic in).

2) Set Radios to VOX or External PTT

  - For CM108-based PTT control, configure the GPIO pin and device path. (Recommended)

  - VOX is the simplest mode and works out of the box on many radios.

3) Launch Application

  - Select your input and output devices.

  - Set your config variables in the GUI or manually in `config.yaml`

  - Click “Start” or use the `--headless` flag in CLI to begin repeater operation.

4) Deploy Anywhere

  - KrakenRelay is lightweight enough for:

    - Raspberry Pi-based go-box repeaters

    - Temporary field stations

    - Full-time repeater installations

    - Multi-site systems with IP audio links

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
🐙 **Stay Connected:** [LinkStack](https://squidconnect.americansquid.com/@americansquid)
