# Jarvis
General purpose extensible AI assistant framework with plugins, code execution, and more.

## Hardware

Can run in CPU only or GPU mode.

### Tested Systems
- Apple Silicion M1 Pro, 10 Core, 16 GB
  
## Model

Any Llama-cpp model in the GGUF format is valid

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

#### Kwargs
Kwargs (keyword arguments) is a json dictionary that is passed into the `Plugin` class's `**kwargs` argument when it's loaded. 

## TODO
 - [ ] Discord interface (WIP)
 - [ ] Image generation (using Koala?)
 - [ ] Docker image
 - [x] Plugins interface
 - [x] Code execution
