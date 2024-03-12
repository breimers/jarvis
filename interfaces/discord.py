"""discord
    permission_int: 274877979648
    token: MTIxNjkxODI3NjE2OTI2OTI4OA.GvSFvE.EM6q-D8Y16sDEr6QepEQ05vWnCS7JPRkUDIRgM
    appid: 1216918276169269288
    inivte_url: https://discordapp.com/oauth2/authorize?client_id=1216918276169269288&scope=bot&permissions=274877979648
"""

import sys
import discord

if __name__ == "__main__":
    args = sys.argv[1:]
    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f'Logged in as {client.user}')

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        if message.content.startswith('@Talos'):
            prompt = str(message.content)
            await message.channel.send('Hello!')

    client.run(args[0])