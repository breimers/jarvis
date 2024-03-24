import json

from chat import ChatBot

if __name__ == "__main__":
    config = json.load(open("config.json"))
    kwargs = config.get("chat_kwargs", dict())
    chat = ChatBot(config=config, **kwargs)
    chat.start_shell()