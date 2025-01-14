# SpeechT5 Assistant

This project demonstrates an end-to-end pipeline for a speech-enabled assistant, leveraging advanced tools such as Microsoft SpeechT5 models and OpenAI's GPT-based API. It performs the following key functionalities:

1. **Text-to-Speech (TTS):** Convert text into a spoken audio file.
2. **Automatic Speech Recognition (ASR):** Transcribe spoken audio back into text.
3. **ChatGPT Integration:** Generate intelligent responses to transcribed text.
4. **Performance Evaluation:** Assess transcription accuracy using Word Error Rate (WER).

---

## Features

### Text-to-Speech (TTS)
Converts input text to a spoken audio file using the Microsoft SpeechT5 TTS model and Hifi-GAN vocoder. The output is saved as a `.wav` file.

### Automatic Speech Recognition (ASR)
Processes audio input and transcribes it into text using the Microsoft SpeechT5 ASR model.

### ChatGPT Integration
Generates contextual responses to user input by leveraging OpenAI’s API. The assistant’s personality is customisable via an external `bot_personality.txt` file.

### Performance Evaluation
Measures the accuracy of the ASR process by calculating the Word Error Rate (WER) between the reference and hypothesis texts.

---

## Setup

### Prerequisites

1. Python 3.8+
2. Required Python libraries:
    - `torch`
    - `transformers`
    - `datasets`
    - `soundfile`
    - `openai`
    - `jiwer`
3. An OpenAI API key saved in `api_key.txt`.
4. A text file `bot_personality.txt` containing the assistant’s personality description.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Add your OpenAI API key to `api_key.txt` and set up `bot_personality.txt` as required.

---

## Usage

### Example Workflow

1. **Text-to-Speech:**
    ```bash
    python main.py
    ```
    Generates speech from a sample text, saves it as `assistant_audio.wav`.

2. **Speech-to-Text:**
    Transcribes the generated audio and prints the transcription.

3. **Evaluate Performance:**
    Compares the transcription with the original text and calculates the Word Error Rate (WER).

4. **ChatGPT Response:**
    Generates a response to the transcribed text.

5. **Response-to-Speech:**
    Converts the ChatGPT response back into audio and saves it as `response_audio.wav`.

---

## File Descriptions

- **main.py:** Core script for the speech assistant pipeline.
- **api_key.txt:** Stores your OpenAI API key (not included, create this file).
- **bot_personality.txt:** Contains personality settings for the assistant.

---

## Example Output

1. **Initial Text:**
    > Why did the chicken cross the road? Because there was food on the other side.

2. **Transcription:**
    > Why did the chicken cross the road? Because there was food on the other side.

3. **ChatGPT Response:**
    > To satisfy its hunger and perhaps its curiosity too!

4. **Performance Evaluation:**
    > Word Error Rate (WER): 0.00

5. **Generated Audio:**
    Saved as `assistant_audio.wav` and `response_audio.wav`.

---

## Notes

- Ensure audio files are in `.wav` format with a 16 kHz sampling rate.
- Modify the `initial_text` in the script for customised interactions.
- Handle errors with care, particularly for API usage and file dependencies.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

