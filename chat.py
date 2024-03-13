"""Jarvis.Chat

"""

# Native imports
import os
import json
import importlib
from datetime import datetime

# LLM Imports
from llama_cpp import Llama

class ChatMessage:
    def __init__(self, actor, content) -> None:
        self.actor = actor
        self.content = content
    
    def dict(self) -> str:
        return {"actor": str(self.actor), "content": str(self.content)}
    
    def json(self) -> str:
        return json.dumps(self.dict())


class ChatHistory:
    def __init__(self, system_message) -> None:
        self.system_message = system_message
        self.messages = list()
        
    def create_prompt(self):
        prompt = f"<|im_start|>system\n{self.system_message}\n<|im_end|>"
        for message in self.messages:
            prompt += f"\n<|im_start|>{message.actor}\n{message.content}\n<|im_end|>"
        prompt += "\n<|im_start|>assistant\n"
        return prompt
    
    def add(self, actor, message):
        self.messages.append(ChatMessage(actor, message))
    
    def save(self, location="history"):
        filename = os.path.join(os.path.abspath(location), f"{str(int(datetime.now().timestamp()))}.txt")
        with open(filename, "w") as f:
            json.dump([m.dict() for m in self.messages], f)
        return filename


class GenerationArgs:
    def __init__(self, max_tokens=512, temperature=1.00, top_k=1, top_p=0.9) -> None:
        self.max_tokens=max_tokens
        self.temperature=temperature
        self.top_k=top_k
        self.top_p=top_p
        
    def dict(self):
        return {
            "max_tokens":self.max_tokens,
            "stop":["<|im_end|>"],
            "echo":False,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }


class IntentionManager:
    def __init__(self, chat_bot, plugin_map="plugins/plugins.json") -> None:
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
        plugin_intents = list()
        for k in self.plugin_map.keys():
            if any([word.lower() in input.lower() for word in self.plugin_map[k]["intents"]]):
                plugin_intents.append(k)
        return plugin_intents
    
    def handle(self, input):
        plugin_intents = self.infer(input)
        for plugin_intent in plugin_intents:
            self.plugin_map[plugin_intent]["obj"].run(input)


class ChatBot:
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
        self.model = Llama(
            model_path=model_path,
            n_ctx=context_length, 
            n_threads=os.cpu_count()-1,
            n_gpu_layers=gpu_layers,
            f16_kv=True,
        )
        
    def call(self, actor, input):
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
