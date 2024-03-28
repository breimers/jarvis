import requests
import json
from .base import Plugin

class SearchEngine(Plugin):
    """Class representing a save plugin."""
    
    def __init__(self, name="SearchEngine", chat_bot=None, **kwargs) -> None:
        """Initialize a Save plugin.

        Args:
            name (str, optional): The name of the plugin. Defaults to "SearchEngine".
            chat_bot (ChatBot, optional): The chatbot instance. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super(SearchEngine, self).__init__(name=name, chat_bot=chat_bot)
        self.intents = ["search", "google"]
        self.results = kwargs.get("results", 1)
        self.cx = kwargs.get("cx", False)
        self.keyfile = kwargs.get("keyfile", False)
        self.url = kwargs.get("url", "https://www.googleapis.com/customsearch/v1")
        self.fields = "items"
        if not all([self.results, self.cx, self.keyfile, self.url]):
            raise AttributeError("Keyfile and CX code must be specified in plugins.json")
        with open(self.keyfile, "r") as kf:
            self.key = kf.read()
        self.params = {
            "cx": self.cx,
            "key": self.key,
            "fields": self.fields,
            "q": "date and time"
        }
        
    def run(self, input):
        """Run the Google SearchEngine plugin.

        Args:
            input: The input text.
        """
        print("***running search***")
        params = self.params.copy()
        params["q"] = ' '.join([token for token in input.split() if token not in self.intents])
        response = requests.get(url=self.url, params=params).json()
        print("***got response, processing results***")
        results = response.get("items", [])[:(int(self.results)-1)]
        results = "\n".join(
            [
                json.dumps({
                    "title": result.get("title", "None"),
                    "link": result.get("link", "None"),
                    "snippet": result.get("snippet", "None")
                }, indent=4) for result in results
            ]
        )
        print("***sending results to llm***")
        self.chat_bot.history.add(
            'google-search', 
            f"Request: {input}\n\nResults JSON ({self.results}):\n{results}\n"
        )
        self.chat_bot.history.add(
            'system',
            "Please format the google-search results into a user friendly format."
        )
