"""Jarvis.Chat

"""

# Native imports
import os
import json
import shlex
import subprocess
from datetime import datetime
from typing import Any, List, Mapping, Optional

# LLM Imports
from llama_cpp import Llama

# RAG Imports
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

# Custom Imports
from plugins import *


DEFAULT_INTENTS = {
    "RAG": ["search", "find", "query", "retrieve", "look", "lookup", "research"],
    "EXEC": ["exec", "execute", "run", "command", "cmd"],
    "SAVE": ["/save", "/keep"]
}


class LangchainWrapper(LLM):
    model_instance: Llama

    @property
    def _llm_type(self) -> str:
        return "llama-cpp"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = ["<|im_end|>"],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return str(self.model_instance(prompt))

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_instance": self.model_instance}


class RAGPipeline:
    def __init__(self, source="./rag/source", llm=None) -> None:
        loader = DirectoryLoader(source)
        docs = loader.load()
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma.from_documents(documents=docs, persist_directory='./rag/vectors', embedding=self.embeddings)
        self.llm = llm
        
    def query(self, query):
        qa = RetrievalQA.from_llm(self.llm, retriever=self.vectorstore.as_retriever())
        results = qa(query)
        answer = results['result']
        return answer
    


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
    def __init__(self, chat_bot, intention_map=None) -> None:
        if intention_map is None:
            self.intention_map = DEFAULT_INTENTS
        else:
            self.intention_map = intention_map
        self.chat_bot = chat_bot

    def infer(self, input):
        discovered_intents = list()
        for k, v in DEFAULT_INTENTS.items():
            if any([word.lower() in input.lower() for word in v]):
                discovered_intents.append(k)
        return discovered_intents
    
    def handle(self, input):
        intents = self.infer(input)
        for intent in intents:
            self.__getattribute__(intent)(input)
    
    #  * Capslocking chat plugin methods to distinguish from normal methods
    #     as these will be moved to their own class at some point.
    #  * These definitely don't belong here, IntentionManager should match 
    #     the intention_map keys to plugins.
    #  * ChatBot should not contain plugin logic either.
    def RAG(self, input):
        if hasattr(self.chat_bot, 'rag'):
            answer = self.chat_bot.rag.query(str(input))
            self.chat_bot.history.add('document-retrieval', answer)
        self.chat_bot.history.add('system', f"Please summarize the document-retrieval results, while keeping in mind the last user message.")

    def EXEC(self, input, timeout=300):
        input = str(input)
        if any([input.startswith(key) for key in self.intention_map["EXEC"]]):
            command = input.split(maxsplit=1)[1]
            self.chat_bot.history.add('executor', f"Running command: {command}\n\n")
            cmd = shlex.split(command, posix=True)
            try:
                proc = subprocess.run(args=cmd, universal_newlines=True, capture_output=True, text=True, timeout=timeout)
                stdout = proc.stdout
                stderr = proc.stderr
                self.chat_bot.history.add('executor', f"Result: {stdout}\nError: {stderr}")
            except subprocess.TimeoutExpired:
                self.chat_bot.history.add('executor', f"Error: System failed to respond within {timeout} seconds.")
            except subprocess.SubprocessError as e:
                self.chat_bot.history.add('executor', f"Error: {e}")
            self.chat_bot.history.add('system', "Please report the executor results for the user.")
            
    def SAVE(self, input):
        _ = input
        filename = self.chat_bot.history.save()
        self.chat_bot.history.add('system', f"Inform user that the current chat history was saved to {filename}.")


class ChatBot:
    def __init__(
        self, 
        system_prompt="You are a helpful AI assistant.",
        temperature=1.00,
        top_p=0.9,
        top_k=1,
        max_tokens=512,
        model_path="/Users/breimers/Workshop/models/llm/dolphin-2.6-mistral-7b-dpo-laser-Q8_0.gguf",
        data_store="rag/source",
        exec=False,
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
        self.intentions = IntentionManager(self)
        # Move to Plugins later
        if data_store is not None:
            self.load_rag_pipeline(data_store)
        # Move to Plugins later
        if exec:
            self.exec = True
    
    # Move to Plugins once the class is written
    def load_rag_pipeline(self, data_store):
        llm_wrapper = LangchainWrapper(model_instance=self.model)
        self.rag = RAGPipeline(data_store, llm_wrapper)
    
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
    chat = ChatBot(exec=True)
    chat.start_shell()
