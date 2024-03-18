# Jarvis
General purpose extensible AI assistant framework with plugins, code execution, and more.

## Getting Started

### Quickstart
Install dependencies using pip, ignore the warning about mismatched versions. 

`pip install -r requirements.txt`

Run a shell session with the assistant using default settings.

`python3 ./chat.py`

### Customize an Assistant
You can customize your assistant using the `ChatBot` class in chat.py:

```python
from chat import ChatBot

my_assistant = ChatBot(
        system_prompt="You are a helpful AI assistant.",
        temperature=1.00,
        top_p=0.9,
        top_k=1,
        max_tokens=512,
        model_path="dolphin-7b-Q8_0.gguf",
        context_length=16000,
        gpu_layers=-1
)
```

## Environnment

### Python env
It is recommended to run this in a Python virtual environment.

### Shell env
Optionally, enable `TOKENIZERS_PARALLELISM` by running `source env.sh`

This will enable parallelism but could cause deadlocks.

## Hardware

Can run in CPU only or GPU (including metal) mode.

### Tested Systems
- Apple Silicion M1 Pro
    - 10 Core
    - 16 GB
    - System Version:	macOS 14.1.1 (23B81)
    - Kernel Version:	Darwin 23.1.0

- Linode Shared CPU VM
    - 4 vCPU
    - 8 GB
    - Ubuntu 22.04 LTS

## Model

Any Llama-cpp model in the GGUF format should work.

### Tested Models
- [Dolphin 2.6 Mistral 7B DPO Laser (Quantized)](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-laser-GGUF)

## Chat 

### Instruction Format

Currently implemented as ChatML only

## Plugins

### Loading plugins
Plugins are loaded via the `plugins/plugins.json` file. 

File names should be the same as the name of the second class element *(e.g. rag, exec, base)*, with a `.py` extension.

### Default Plugin

Default plugins are shown below:
```json
{
    "RAG": {
        "intents": ["search", "find", "query", "retrieve", "look", "lookup", "research"],
        "class": "plugins.rag.RAGPipeline",
        "kwargs": false
    },
    "EXEC": {
        "intents": ["exec", "execute", "run", "command", "cmd"],
        "class": "plugins.exec.Executor",
        "kwargs": {"timeout": 300}
    },
    "SAVE": {
        "intents": ["/save", "/keep"],
        "class": "plugins.base.Save",
        "kwargs": false
    }
}
```

### Plugin Development

#### Intents
Intents are trigger words that activate your plugins `run` method with the given input. 

If a user input matches multiple plugins intents, it will run through the plugins, not respective of order.

#### Class
Class is a reference to the `Plugin` class for the given plugin.

The naming convention is as follows: `plugins.<plugin-file.py>.<Plugin class>`

`Plugin` classes should always have an `__init__`, and a `run` method. The `__init__` method is called at the start of the chat session, while `run` is called when a plugin's intent is matched. 

*plugins/myplugin.py*
```python
from .base import Plugin

class MyPlugin(Plugin):
    def __init__(self, name="MyPlugin", chat_bot=None, **kwargs) -> None:
        super(MyPlugin, self).__init__(name=name, chat_bot=chat_bot)
        if kwargs.get("myvar"):
            self.var = kwargs.get("myvar")
        else:
            self.var= "hello"

    def my_function(self, x):
        return x[::-1]

    def run(self, input):
        ## Perform logic on input
        reversed_input = self.my_function(input)
        ## Update chat history
        self.chat_bot.history.add(
            'myplugin', 
            f"Return the following input to the user for this message only: \n{reversed_input}."
        )
```
*plugins/plugins.json*
```json
{
    "MYPLUGIN": {
        "intents": ["trigger"],
        "class": "plugins.myplugin.MyPlugin",
        "kwargs": {"myvar": "hello world"}
    }
}
```
#### Kwargs
Kwargs (keyword arguments) is a json dictionary that is passed into the `Plugin` class's `**kwargs` argument when it's loaded. 

## TODO
 - [ ] *Discord interface (WIP)*
 - [ ] Docker image
 - [x] Plugins interface
 - [x] Code execution
 - [x] Koala image generation ()
