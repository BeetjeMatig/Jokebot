import os
import torch
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
from openai import OpenAI
from jiwer import wer

# Read API key from an external file
with open("api_key.txt", "r") as f:
    api_key = f.read().strip()

# Read personality instruction from an external file
with open("bot_personality.txt", "r") as f:
    bot_personality = f.read().strip()

# Set up OpenAI instance
openai = OpenAI(api_key=api_key)

# Step 1: Text-to-Speech (TTS)
def text_to_speech(text, output_path="tts_example.wav"):
    # Load TTS Processor and Model
    tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    # Preprocess text input
    inputs = tts_processor(text=text, return_tensors="pt")

    # Load Speaker Embedding
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7305]["xvector"]).unsqueeze(0)

    # Load Vocoder
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Generate Speech
    speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # Save to File
    sf.write(output_path, speech.numpy(), samplerate=16000)
    print(f"TTS output saved to {output_path}")

# Step 2: Automatic Speech Recognition (ASR)
def speech_to_text_from_file(audio_path):
    # Load ASR Processor and Model
    asr_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
    asr_model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

    # Load audio data from file
    audio_data, _ = sf.read(audio_path)
    inputs = asr_processor(audio=audio_data, sampling_rate=16000, return_tensors="pt")

    # Generate transcription
    predicted_ids = asr_model.generate(**inputs, max_length=100)
    transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print("Transcription:", transcription[0])
    return transcription[0]

# Step 3: Generate Response with ChatGPT
def get_chatgpt_response(prompt):
    try:
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": bot_personality},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating ChatGPT response: {e}")
        return "I'm sorry, I couldn't generate a response."

# Step 4: Evaluate Model Performance
def evaluate_performance(reference_text, hypothesis_text):
    error_rate = wer(reference_text, hypothesis_text)
    print(f"Word Error Rate (WER): {error_rate:.2f}")
    return error_rate

# Example Usage
if __name__ == "__main__":
    # Step 1: Generate audio from text
    initial_text = "Why did the chicken cross the road? Because there was food on the other side."
    tts_output_path = "assistant_audio.wav"
    text_to_speech(initial_text, tts_output_path)

    # Step 2: Transcribe the generated audio back to text
    transcription = speech_to_text_from_file(tts_output_path)
    print("User Input:", transcription)

    # Step 3: Evaluate ASR Performance
    evaluate_performance(initial_text.lower(), transcription.lower())

    # Step 4: Get a response from ChatGPT
    chatgpt_response = get_chatgpt_response(transcription)
    print("ChatGPT Response:", chatgpt_response)

    # Step 5: Convert the ChatGPT response to speech
    response_audio_path = "response_audio.wav"
    text_to_speech(chatgpt_response, response_audio_path)
    print(f"Response audio saved to {response_audio_path}")
