
from typing import Any, List, Mapping, Optional

from llama_cpp import Llama
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from .base import Plugin

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


class RAGPipeline(Plugin):
    def __init__(self, name="RAG", chat_bot=None, **kwargs) -> None:
        super(RAGPipeline, self).__init__(name=name, chat_bot=chat_bot)
        if kwargs.get("source"):
            self.source = kwargs.get("source")
        else:
            self.source = "rag/source"
        loader = DirectoryLoader(self.source)
        docs = loader.load()
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma.from_documents(
            documents=docs, 
            persist_directory='./rag/vectors', 
            embedding=self.embeddings
        )
        self.llm = LangchainWrapper(model_instance=self.chat_bot.model)
        self.qa = RetrievalQA.from_llm(self.llm, retriever=self.vectorstore.as_retriever())

        
    def run(self, input):
        results = self.qa(input)
        answer = results['result']
        self.chat_bot.history.add('document-retrieval', answer)