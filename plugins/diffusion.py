"""Diffuser Plugin

This module defines a plugin for running Diffussion image generation models.
"""
import os
from .base import Plugin
from diffusers import DiffusionPipeline
from datetime import datetime

import torch

class Diffuser(Plugin):
    """Class representing an Diffuser plugin."""
    
    def __init__(self, name="DIFF", chat_bot=None, **kwargs) -> None:
        """Initialize an Diffuser plugin.

        Args:
            name (str, optional): The name of the plugin. Defaults to "DIFF".
            chat_bot (ChatBot, optional): The chatbot instance. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super(Diffuser, self).__init__(name=name, chat_bot=chat_bot)
        self.intents = ["generate", "render", "draw", "sketch"]
        self.device = kwargs.get('device', "cuda")
        if kwargs.get('destination'):
            self.destination = os.path.abspath(kwargs.get('destination'))
        else:
            self.destination = None
        self.show = kwargs.get('show', True)
        self.model = self.load_model(kwargs.get('model_path', None))            
            
    def load_model(self, model_path):
        if not model_path:
            model_path="etri-vilab/koala-700m"
        if self.device == "cpu":
            pipe = DiffusionPipeline.from_pretrained(model_path)
        else:
            pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe = pipe.to(self.device)
        pipe.enable_attention_slicing()
        return pipe

    def run(self, input):
        """Run the Diffuser plugin."""
        try:
            image = self.model(input).images[0]
            self.chat_bot.history.add('diffusion-artist', f"Successfully generated an image from the prompt: \n'{input}'\n")
            if self.destination:
                filename = f"{str(int(datetime.now().timestamp()))}.png"
                filepath = os.path.join(self.destination, filename)
                image.save(filepath)
                self.chat_bot.history.add('diffusion-artist', f"Saved image to {self.destination}")
            if self.show:
                image.show()
        except Exception as e:
            self.chat_bot.history.add('diffusion-artist', f"Diffuser Plugin failed with the following error: \n'{e}'\n")
        self.chat_bot.history.add('system', f"Do not respond directly to the user's last request, instead you must report the information from diffusion-artist.")