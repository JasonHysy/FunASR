from funasr import AutoModel
import sounddevice as sd
import numpy as np

chunk_size = [0, 10, 5]  # Configure as needed, e.g., [0, 10, 5] for 600ms, [0, 8, 4] for 480ms
encoder_chunk_look_back = 4  # Number of chunks to look back for encoder self-attention
decoder_chunk_look_back = 1  # Number of encoder chunks to look back for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming")

sample_rate = 16000  # Most common sampling rate used
chunk_duration = 0.6  # Duration of each chunk in seconds
chunk_stride = int(sample_rate * chunk_duration)  # Convert duration to number of samples

cache = {}
# Choose the device explicitly, either 0 or 1, based on your needs
# device_id = 'pulse'

def callback(indata, frames, time, status):
    if status:
        print(status)
    speech_chunk = indata[:, 0]  # Assuming mono channel input
    is_final = len(speech_chunk) < chunk_stride
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
    print(res)

with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback, blocksize=chunk_stride):
    print("Starting real-time speech recognition. Press Ctrl+C to stop.")
    sd.sleep(-1)  # Wait indefinitely until the user stops the execution
