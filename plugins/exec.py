"""Executor Plugin

This module defines a plugin for executing commands.
"""

import shlex
import subprocess

from .base import Plugin


class Executor(Plugin):
    """Class representing an executor plugin."""
    
    def __init__(self, name="EXEC", chat_bot=None, **kwargs) -> None:
        """Initialize an Executor plugin.

        Args:
            name (str, optional): The name of the plugin. Defaults to "EXEC".
            chat_bot (ChatBot, optional): The chatbot instance. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super(Executor, self).__init__(name=name, chat_bot=chat_bot)
        self.timeout = 300
        self.intents = ["exec", "execute", "run", "command", "cmd"]

        if kwargs.get('timeout'):
            self.timeout = int(kwargs.get('timeout'))
            
    def run(self, input):
        """Run the executor plugin.

        Args:
            input (str): The input text.
        """
        input = str(input)            
        for key in self.intents:
            if key in input:
                command = input.split(f"{key} ", maxsplit=1)[1]
                self.chat_bot.history.add(
                    'executor', 
                    f"Running command: {command}\n\n"
                )
                cmd = shlex.split(command, posix=True)
                try:
                    proc = subprocess.run(
                        args=cmd, 
                        universal_newlines=True, 
                        capture_output=True, 
                        text=True, 
                        timeout=self.timeout
                    )
                    stdout = proc.stdout
                    stderr = proc.stderr
                    self.chat_bot.history.add('executor', f"Result: {stdout}\nError: {stderr}")
                except subprocess.TimeoutExpired:
                    self.chat_bot.history.add(
                        'executor', 
                        f"Error: System failed to respond within {self.timeout} seconds."
                    )
                except subprocess.SubprocessError as e:
                    self.chat_bot.history.add('executor', f"Error: {e}")
                self.chat_bot.history.add(
                    'system', 
                    "Please report the executor results for the user."
                )
                break
            else:
                continue
