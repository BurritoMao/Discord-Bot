import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv
import os
import struct
import asyncio

load_dotenv()
token = os.getenv("DISCORD_TOKEN")

handler = logging.FileHandler(filename="discord.log", encoding="utf-8", mode="w")
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} - {bot.user.id}")
    print("------")


@bot.event
async def on_connect():
    print("Bot connected to Discord!")


# Helper Functions
async def ask_server(host, port, data_bytes):
    """Helper function to handle the socket connection and framing."""
    reader, writer = await asyncio.open_connection(host, port)

    # Send length header + data
    writer.write(struct.pack(">I", len(data_bytes)))
    writer.write(data_bytes)
    await writer.drain()

    # Receive response length header + data
    length_bytes = await reader.readexactly(4)
    resp_length = struct.unpack(">I", length_bytes)[0]
    response_data = await reader.readexactly(resp_length)

    writer.close()
    await writer.wait_closed()
    return response_data


# Bot Commands
@bot.command()
async def talk(ctx, *, user_input: str):
    """Takes text, passes it through the LLM and TTS pipelines via sockets."""
    if not ctx.author.voice:
        return await ctx.send("Join a voice channel first!")

    # 1. Connect to voice
    vc = ctx.voice_client
    if not vc:
        vc = await ctx.author.voice.channel.connect()

    await ctx.send("Thinking...")

    try:
        # 2. Send text to LLM socket (Port 5001)
        llm_response_bytes = await ask_server(
            "127.0.0.1", 5001, user_input.encode("utf-8")
        )
        llm_text = llm_response_bytes.decode("utf-8")
        await ctx.send(f"**AI says:** {llm_text}")

        # 3. Send LLM text to TTS socket (Port 5002)
        audio_bytes = await ask_server("127.0.0.1", 5002, llm_response_bytes)

        # 4. Save bytes to a temp file and play
        temp_path = "temp_output.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        if not vc.is_playing():
            vc.play(
                discord.FFmpegPCMAudio(temp_path),
                after=lambda err: (
                    os.remove(temp_path) if os.path.exists(temp_path) else None
                ),
            )

    except ConnectionRefusedError:
        await ctx.send("Error: Make sure both the LLM and TTS servers are running!")


logger = logging.getLogger("discord")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Run the bot with just the token
bot.run(token)
