# Qwen3-TTS-IREE (doesn't work)

## Getting Started

To get this project running on your local machine, follow the steps below. I recommend using a virtual environment to keep your global Python installation clean and avoid dependency conflicts.

### Installation

1. **Clone the repository**
Open your terminal and run the following command to clone the project:

```bash
git clone https://github.com/i448/qwen3-tts-iree
cd qwen3-tts-iree

```

1. **Set up a Virtual Environment (Recommended)**
I suggest using `venv` to manage the project dependencies. Run these commands to create and activate your environment:

```bash
python -m venv .venv
source .venv/bin/activate

```

> I personally use Linux, and I don't use Windows so you have to search it up. 🙃

1. **Install dependencies**
With your virtual environment active, install the required packages using the provided requirements file:

```bash
pip install -r requirements.txt

```

### Running the Application

Once the installation is complete, you can start the program by executing the main script _(which will fail)_:

```bash
python jit2/main.py

```

[Notes](NOTES.md)