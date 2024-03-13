"""Jarvis.Chat

This module defines classes for managing chat messages, chat history, generation arguments, intention management, and the chatbot itself.
"""

# Native imports
import os
import json
import importlib
from datetime import datetime

# LLM Imports
from llama_cpp import Llama


class ChatMessage:
    """Class representing a chat message."""
    
    def __init__(self, actor, content) -> None:
        """Initialize a ChatMessage object.

        Args:
            actor (str): The actor who sent the message.
            content (str): The content of the message.
        """
        self.actor = actor
        self.content = content
    
    def dict(self) -> dict:
        """Convert the message to a dictionary.

        Returns:
            dict: The message represented as a dictionary.
        """
        return {"actor": str(self.actor), "content": str(self.content)}
    
    def json(self) -> str:
        """Convert the message to JSON format.

        Returns:
            str: The message represented in JSON format.
        """
        return json.dumps(self.dict())


class ChatHistory:
    """Class for managing chat history."""
    
    def __init__(self, system_message) -> None:
        """Initialize a ChatHistory object.

        Args:
            system_message (str): The system message for the chat history.
        """
        self.system_message = system_message
        self.messages = list()
        
    def create_prompt(self):
        """Create a prompt including system message and chat messages.

        Returns:
            str: The prompt including system message and chat messages.
        """
        prompt = f"<|im_start|>system\n{self.system_message}\n<|im_end|>"
        for message in self.messages:
            prompt += f"\n<|im_start|>{message.actor}\n{message.content}\n<|im_end|>"
        prompt += "\n<|im_start|>assistant\n"
        return prompt
    
    def add(self, actor, message):
        """Add a message to the chat history.

        Args:
            actor (str): The actor who sent the message.
            message (str): The content of the message.
        """
        self.messages.append(ChatMessage(actor, message))
    
    def save(self, location="history"):
        """Save the chat history to a file.

        Args:
            location (str, optional): The location to save the chat history. Defaults to "history".

        Returns:
            str: The filename of the saved chat history.
        """
        filename = os.path.join(os.path.abspath(location), f"{str(int(datetime.now().timestamp()))}.txt")
        with open(filename, "w") as f:
            json.dump([m.dict() for m in self.messages], f)
        return filename


class GenerationArgs:
    """Class for managing generation arguments."""
    
    def __init__(self, max_tokens=512, temperature=1.00, top_k=1, top_p=0.9) -> None:
        """Initialize a GenerationArgs object.

        Args:
            max_tokens (int, optional): The maximum number of tokens for generation. Defaults to 512.
            temperature (float, optional): The temperature for generation. Defaults to 1.00.
            top_k (int, optional): The top-k value for generation. Defaults to 1.
            top_p (float, optional): The top-p value for generation. Defaults to 0.9.
        """
        self.max_tokens=max_tokens
        self.temperature=temperature
        self.top_k=top_k
        self.top_p=top_p
        
    def dict(self):
        """Convert the generation arguments to a dictionary.

        Returns:
            dict: The generation arguments represented as a dictionary.
        """
        return {
            "max_tokens":self.max_tokens,
            "stop":["<|im_end|>"],
            "echo":False,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }


class IntentionManager:
    """Class for managing intentions."""
    
    def __init__(self, chat_bot, plugin_map="plugins/plugins.json") -> None:
        """Initialize an IntentionManager object.

        Args:
            chat_bot (ChatBot): The chatbot instance.
            plugin_map (str, optional): The path to the plugin map JSON file. Defaults to "plugins/plugins.json".
        """
        self.chat_bot = chat_bot
        with open(plugin_map) as jsonf:
            self.plugin_map = json.load(jsonf)
        for k, v in self.plugin_map.items():
            module_name, class_name = v['class'].rsplit('.', 1)
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            kwargs = v["kwargs"] if isinstance(v["kwargs"], dict) else {}
            self.plugin_map[k]["obj"] = class_(chat_bot=chat_bot, **kwargs)

    def infer(self, input):
        """Infer the intentions based on the input.

        Args:
            input (str): The input text.

        Returns:
            list: A list of inferred intentions.
        """
        plugin_intents = list()
        for k in self.plugin_map.keys():
            if any([word.lower() in input.lower() for word in self.plugin_map[k]["intents"]]):
                plugin_intents.append(k)
        return plugin_intents
    
    def handle(self, input):
        """Handle the input based on inferred intentions.

        Args:
            input (str): The input text.
        """
        plugin_intents = self.infer(input)
        for plugin_intent in plugin_intents:
            self.plugin_map[plugin_intent]["obj"].run(input)


class ChatBot:
    """Class representing a chatbot."""
    
    def __init__(
        self, 
        system_prompt="You are a helpful AI assistant.",
        temperature=1.00,
        top_p=0.9,
        top_k=1,
        max_tokens=512,
        model_path="/Users/breimers/Workshop/models/llm/dolphin-2.6-mistral-7b-dpo-laser-Q8_0.gguf",
        context_length=16000,
        gpu_layers=-1
    ) -> None:
        """Initialize a ChatBot object.

        Args:
            system_prompt (str, optional): The system prompt for the chatbot. Defaults to "You are a helpful AI assistant.".
            temperature (float, optional): The temperature for generation. Defaults to 1.00.
            top_p (float, optional): The top-p value for generation. Defaults to 0.9.
            top_k (int, optional): The top-k value for generation. Defaults to 1.
            max_tokens (int, optional): The maximum number of tokens for generation. Defaults to 512.
            model_path (str, optional): The path to the model. Defaults to "/Users/breimers/Workshop/models/llm/dolphin-2.6-mistral-7b-dpo-laser-Q8_0.gguf".
            context_length (int, optional): The context length for generation. Defaults to 16000.
            gpu_layers (int, optional): The number of GPU layers. Defaults to -1.
        """
        self.history = ChatHistory(system_prompt)
        self.gen_args = GenerationArgs(
            max_tokens, 
            temperature, 
            top_k, 
            top_p
        )
        self.load_model(model_path, context_length, gpu_layers)
        self.intentions = IntentionManager(chat_bot=self)
    
    # Move to Plugins once the class is written
    
    def load_model(self, model_path, context_length, gpu_layers):
        """Load the model.

        Args:
            model_path (str): The path to the model.
            context_length (int): The context length.
            gpu_layers (int): The number of GPU layers.
        """
        self.model = Llama(
            model_path=model_path,
            n_ctx=context_length, 
            n_threads=os.cpu_count()-1,
            n_gpu_layers=gpu_layers,
            f16_kv=True,
        )
        
    def call(self, actor, input):
        """Call the chatbot with user input.

        Args:
            actor (str): The actor who sent the input.
            input (str): The input text.

        Returns:
            str: The response from the chatbot.
        """
        self.history.add(actor, input)
        self.intentions.handle(input)
        res = self.model(
            self.history.create_prompt(), 
            **self.gen_args.dict()
        )
        text_response = res["choices"][0]["text"]
        self.history.add("assistant", text_response)
        return text_response

    def start_shell(self):
        """Start the interactive shell for the chatbot."""
        while True:
            try:
                user_input = str(input("Enter Text >>  "))
                assistant_response = self.call("user", user_input)
                print(f"Assistant << {assistant_response}")
            except KeyboardInterrupt:
                self.history.save()
                break


if __name__ == "__main__":
    chat = ChatBot()
    chat.start_shell()