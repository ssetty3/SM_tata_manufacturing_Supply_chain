# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
from agent_graph import run_finance_agent, AgentConfig

app = FastAPI(title="Finance Agent (LangGraph)")

class ChatRequest(BaseModel):
    user_id: str = Field(..., example="user-123")
    session_id: str = Field(..., example="sess-1")
    query: str = Field(..., example="Summarize market risk in the documents")
    role: Literal["analyst", "scientist", "financial"] = "analyst"
    retriever_k: int = 4
    summary_every_n_turns: int = 5
    enable_web_fallback: bool = True

class ChatResponse(BaseModel):
    answer: str
    trace: list
    sources: list
    session: Dict[str, Any]

@app.post("/chat_agent", response_model=ChatResponse)
def chat_agent(req: ChatRequest):
    try:
        cfg = AgentConfig(
            retriever_k=req.retriever_k,
            summary_every_n_turns=req.summary_every_n_turns,
            enable_web_fallback=req.enable_web_fallback,
        )
        result = run_finance_agent(
            user_id=req.user_id,
            session_id=req.session_id,
            query=req.query,
            role=req.role,
            config=cfg,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
