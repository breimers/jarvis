"""Plugin Base

This module defines a base class for plugins.
"""

class Plugin:
    """Base class for plugins."""
    
    def __init__(self, name, chat_bot=None, **kwargs) -> None:
        """Initialize a Plugin object.

        Args:
            name (str): The name of the plugin.
            chat_bot (ChatBot, optional): The chatbot instance. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.chat_bot = chat_bot

    def run(self, input):
        """Run the plugin's logic. 
        You should overwrite this function when creating a subclass.

        Args:
            input: The input text.
        """
        self.chat_bot.history.add(
            'plugin-agent', 
            f"User sent >> {input}."
        )

        
class Save(Plugin):
    """Class representing a save plugin."""
    
    def __init__(self, name="Save", chat_bot=None, **kwargs) -> None:
        """Initialize a Save plugin.

        Args:
            name (str, optional): The name of the plugin. Defaults to "Save".
            chat_bot (ChatBot, optional): The chatbot instance. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super(Save, self).__init__(name=name, chat_bot=chat_bot)
        if kwargs.get("location"):
            self.location = kwargs.get("location")
        else:
            self.location= "history"

    def run(self, input):
        """Run the save plugin.

        Args:
            input: The input text.
        """
        _ = input
        filename = self.chat_bot.history.save(location=self.location)
        self.chat_bot.history.add(
            'system', 
            f"Inform user that the current chat history was saved to {filename}."
        )
