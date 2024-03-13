# Jarvis

## Hardware

Can run in CPU only or GPU mode.

### Tested Systems
- Apple Silicion M1 Pro, 10 Core, 16 GB
  
## Model

Any Llama-cpp model in the GGUF format is valid

## Chat Instruction Format

Currently implemented as ChatML only

## RAG

To use RAG, simply add items to [rag/source](./rag/source) and ensure datastore path in ChatBot object is correct. 
If datastore variable in ChatBot is set, it will attempt to build a vector store.

## TODO
 - [ ] Discord interface (WIP)
 - [ ] Code execution
 - [ ] Image generation (using Koala?)
 - [ ] Docker image
