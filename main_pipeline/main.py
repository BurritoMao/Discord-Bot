from indextts.infer import IndexTTS
import asyncio
import struct
import io
import scipy.io.wavfile as wavfile
from io import BytesIO
import numpy as np

# Ensure config.yaml is present in the checkpoints directory
tts = IndexTTS(
    model_dir=r"C:\Users\Burrito\Documents\Discord_Bot\index-tts-lora\finetune_models",
    cfg_path=r"C:\Users\Burrito\Documents\Discord_Bot\index-tts-lora\finetune_models\config_finetuned.yaml",
)

voice = (
    r"C:\Users\Burrito\Documents\Discord_Bot\index-tts-lora\Audio_data\chunk_0000.wav"
)


def debug_wav_type():
    sampling_rate, wav_numpy_array = tts.infer(
        voice, "debug type test", output_path=None
    )
    wav_bytes_io = io.BytesIO()
    wavfile.write(wav_bytes_io, sampling_rate, wav_numpy_array)
    wav_bytes = wav_bytes_io.getvalue()
    with open("debug.wav", "wb") as f:
        f.write(wav_bytes)

    sr, data = wavfile.read(BytesIO(wav_bytes))
    print(sr, data.shape, data.dtype)
    print("max:", wav_numpy_array.max(), "min:", wav_numpy_array.min())
    print("nonzero count:", np.count_nonzero(wav_numpy_array))
    print("slice 1000:1010:", wav_numpy_array[1000:1010])
    print(
        "max",
        wav_numpy_array.max(),
        "min",
        wav_numpy_array.min(),
        "count",
        np.count_nonzero(wav_numpy_array),
    )


async def handle_client(reader, writer):
    # Read the length of the incoming message (4 bytes)
    length_bytes = await reader.readexactly(4)
    message_length = struct.unpack(">I", length_bytes)[0]

    # Read the actual message based on the length
    message_bytes = await reader.readexactly(message_length)
    user_text = message_bytes.decode("utf-8")
    print(user_text)

    # Generate TTS audio
    sampling_rate, wav_numpy_array = tts.infer(voice, user_text, output_path=None)

    # Convert numpy array to bytes
    wav_bytes_io = io.BytesIO()
    wavfile.write(wav_bytes_io, sampling_rate, wav_numpy_array)
    wav_bytes = wav_bytes_io.getvalue()

    # Send the response back to the client
    writer.write(struct.pack(">I", len(wav_bytes)))  # Send length first
    writer.write(wav_bytes)  # Then send the actual audio data
    await writer.drain()

    writer.close()


async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", 5002)
    print("🗣️ TTS Server listening on port 5002...")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    # debug_wav_type()
    asyncio.run(main())
