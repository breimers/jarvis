"""Plugin Base
"""

class Plugin:
    def __init__(self, name, chat_bot=None, **kwargs) -> None:
        self.name = name
        self.chat_bot = chat_bot
        
class Save(Plugin):
    def __init__(self, name="Save", chat_bot=None, **kwargs) -> None:
        super(Save, self).__init__(name=name, chat_bot=chat_bot)
        if kwargs.get("location"):
            self.location = kwargs.get("location")
        else:
            self.location= "history"

    def run(self, input):
        _ = input
        filename = self.chat_bot.history.save(location=self.location)
        self.chat_bot.history.add(
            'system', 
            f"Inform user that the current chat history was saved to {filename}."
        )
