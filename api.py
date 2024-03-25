import json

import uvicorn

from chat import ChatBot

from fastapi import Request, FastAPI


app = FastAPI()
config = json.load(open("config.json"))
kwargs = config.get("chat_kwargs", dict())
cb = ChatBot(config=config, **kwargs)


@app.get("/")
def read_root():
    return "Connected."


@app.post("/chat")
async def chat(request: Request):
    prompt = await request.json()
    return {
        "actor": "assistant",
        "content": cb.call(
            actor=prompt.get("actor", "user"), 
            input=prompt.get("input", "[ No Input ]")
        )
    }

uvicorn.run(app, host="127.0.0.1", port=8080)