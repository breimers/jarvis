"""Plugin Base
"""

class Plugin:
    def __init__(self, name) -> None:
        self.name = name
        
class Save(Plugin):
    def run(self, input):
        _ = input
        filename = self.chat_bot.history.save()
        self.chat_bot.history.add('system', f"Inform user that the current chat history was saved to {filename}.")
