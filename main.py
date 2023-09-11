import streamlit as st
import sounddevice as sd 
from scipy.io.wavfile import write
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import openai

api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = api_key

st.set_page_config(
    page_title="Note Taker",
    page_icon=":memo:",
)

# generate prompt and a semantic function for note taker
prompt = "You are a note taker." +\
            "Create advanced bullet-point notes summarizing the important parts of the transcript." +\
                            "Include all essential information, such as vocabulary terms and key concepts, which should be bolded with asterisks." +\
                            "Remove any extraneous language, focusing only on the critical aspects of the passage or topic." +\
                            "Strictly base your notes on the provided information, without adding any external information." +\
                            "Conclude your notes with [End of Notes] to indicate completion." +\
                            "You are taking notes on the following text: {{$input}}"
kernel = sk.Kernel()
kernel.add_text_completion_service("note_Service", OpenAIChatCompletion("gpt-4", api_key=api_key))
note_helper_func = kernel.create_semantic_function(prompt, max_tokens=1000)

audio_tab, text_tab = st.tabs(["Audio", "Text"])

with audio_tab:
    started = st.button("start recording")

    if started:
        st.write("Recording...")
        freq = 44100
        duration = 30
        recording = sd.rec(int(duration * freq),
                    samplerate=freq, channels=1)
        sd.wait()
        write("my_audio.wav", freq, recording)
        st.write("Recording finished")
    else:
        st.write("Press the button to start recording")
    audio_file = open('my_audio.wav', 'rb')
    if audio_file:
        st.write("Audio file recorded - ", audio_file.name)
        st.audio(audio_file)

    #Let's transcribe the audio and use gpt to summarize the notes
    with open("my_audio.wav", "rb") as audio_file:
        audio_transcript = openai.Audio.transcribe("whisper-1", audio_file).text
        st.write("Transcript: ", audio_transcript)
        # call the semantic function with the transcript as the input
        notes = note_helper_func(audio_transcript)                   
        st.write(notes)

with text_tab:
    text_transcript =st.text_area("Enter transcript text")
    if text_transcript:
        notes = note_helper_func(text_transcript)
        st.write(notes)