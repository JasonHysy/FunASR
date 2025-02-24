from funasr import AutoModel
import sounddevice as sd
import numpy as np
import subprocess


def list_devices():
    # List all available audio devices and their IDs
    print("Available audio devices:")
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        print(f"{idx}: {device['name']} (Input Channels: {device['max_input_channels']})")

def choose_device():
    # Display the list of devices
    list_devices()
    
    # Ask the user to choose a device
    while True:
        try:
            device_id = int(input("Please enter the device ID you want to use: "))
            # Validate if the chosen device ID is within the range of available devices
            if device_id >= 0 and device_id < len(sd.query_devices()):
                # Optional: You can further check if the chosen device has input channels if needed
                if sd.query_devices(device_id)['max_input_channels'] > 0:
                    print(f"\n ==================================== \n You have selected: {sd.query_devices(device_id)['name']} \
                    \n ==================================== \n")
                    return device_id
                else:
                    print("Selected device has no input channels. Please choose another device.")
            else:
                print("Invalid device ID. Please try again.")
        except ValueError:
            print("Please enter a valid number for the device ID.")


device_id = choose_device()
print(f"Device ID {device_id} selected for use.")



chunk_size = [0, 10, 5]  # Configure as needed, e.g., [0, 10, 5] for 600ms, [0, 8, 4] for 480ms
encoder_chunk_look_back = 4  # Number of chunks to look back for encoder self-attention
decoder_chunk_look_back = 1  # Number of encoder chunks to look back for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming")

sample_rate = 16000  # Most common sampling rate used
chunk_duration = 3  # Duration of each chunk in seconds
chunk_stride = int(sample_rate * chunk_duration)  # Convert duration to number of samples

cache = {}
# Choose the device explicitly, either 0 or 1, based on your needs
# device_id = 'pulse'


def callback(indata, frames, time, status):
    if status:
        print("Status: ", status)
    print("Audio data shape: ", indata.shape)  # Check the shape of the incoming audio data
    if np.any(indata):
        print("Data received")  # Verify that non-zero data is coming in
    else:
        print("No data or zero data received")

    speech_chunk = indata[:, 0]
    is_final = len(speech_chunk) < chunk_stride
    res = model.generate(input=speech_chunk, cache=cache, is_final=not is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
    # print("Recognition result: ", res)
    print(res[0]['text'])
    print("run \"screen -S outputSession\" to stream the result on terminal")
    subprocess.run(['screen', '-S', 'outputSession', '-X', 'stuff', res[0]['text']])


try:
    
    with sd.InputStream(samplerate=sample_rate, channels=1, device=device_id, dtype='float32', callback=callback, blocksize=chunk_stride):
        print("Starting real-time speech recognition. Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)  # Keep sleeping in small intervals to keep the loop running
except KeyboardInterrupt:
    print("Stopping...")
except Exception as e:
    print("An error occurred:", e)