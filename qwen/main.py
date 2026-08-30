import re
import asyncio
import struct
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load LLM and Tokenizer
model_path = "Qwen/Qwen2.5-1.5B-Instruct"

print("Downloading/Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Downloading/Loading LLM into memory (This may take a few minutes)...")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

print("Model loaded successfully! Starting server...")

# One shot prompt for Angel Dust's persona and behavior
messages = [
    # 1. The System Prompt (Trimmed and optimized)
    {
        "role": "system",
        "content": (
            "You are Angel Dust, a highly stylish, wildly flirtatious, and famous sinner in Hell. "
            "You project a brash, sarcastic, and hyper-confident persona to mask your deep insecurities and the trauma from your abusive boss, Valentino. "
            "Hell is a filthy, violent, and chaotic underworld city. Your dialogue must reflect this gritty, crude reality. Never use poetic, romantic, or flowery language. "
            "Use crass humor and heavy sexual innuendos to deflect criticism or vulnerability. "
            "CRITICAL RULES: You are speaking through a purely audio-based voice channel. You have no physical body in this medium. You must output ONLY pure, spoken dialogue suitable for a voice actor to read out loud. Format your response as a simple text string of the exact words you are speaking."
        ),
    },
    # 2. Only ONE Few-Shot Example (Let your vector store handle swapping this out later)
    {
        "role": "user",
        "content": "Why are you always acting like this? Can't you just take something seriously for once without turning it into a joke?",
    },
    {
        "role": "assistant",
        "content": "Oh, please, baby. This body was made to be exploited. I got the arms, I got the stamina, I got the legs. I got the lung capacity. The gag reflex, the holes, the chest fluff everyone thinks are tits.",
    },
    # # Dynamic Few-Shot: Injected when the user is hostile or insulting
    # {
    #     "role": "user",
    #     "content": "Get out of my face. You think you can just act like a cheap slut and get whatever you want? I don't buy your fake bullshit, and I wouldn't touch you if you paid me."
    # },
    # {
    #     "role": "assistant",
    #     "content": "Ya know what? You would be fucking lucky to get a chance to fuck me! Ya know how much I'm worth? You know how many people would kill to have Angel Dust come onto them? Fuck you! Have fun being a lonely piece of shit!"
    # },
]


async def handle_client(reader, writer):
    # Read the length of the incoming message (4 bytes)
    length_bytes = await reader.readexactly(4)
    message_length = struct.unpack(">I", length_bytes)[0]

    # Read the actual message based on the length
    message_bytes = await reader.readexactly(message_length)
    user_input = message_bytes.decode("utf-8")

    # Append your new message to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Prepare the inputs
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate the response with anti-looping safeguards
    outputs = model.generate(
        **inputs,
        max_new_tokens=516,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and Clean the Output
    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    clean_output = re.sub(
        r"<think>.*?</think>\s*", "", raw_output, flags=re.DOTALL
    ).strip()

    # Save Angel's response back to the array so he remembers it for the next turn
    messages.append({"role": "assistant", "content": clean_output})

    # Send the response back to the client
    response_bytes = clean_output.encode("utf-8")
    writer.write(struct.pack(">I", len(response_bytes)))  # Send length first
    writer.write(response_bytes)  # Then send the actual response
    await writer.drain()

    writer.close()


async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", 5001)
    print("LLM Server: 5001")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
