from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="React Flow Workflow Backend (Groq)",
    description="Executes React Flow graphs with Groq LLM calls",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class NodeData(BaseModel):
    inputName: Optional[str] = None
    outputName: Optional[str] = None
    text: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    userPrompt: Optional[str] = None
    systemPrompt: Optional[str] = None
    transformType: Optional[str] = None
    channel: Optional[str] = None
    messagePreview: Optional[str] = None
    gateType: Optional[str] = None

class Node(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: NodeData = NodeData()

class Edge(BaseModel):
    id: Optional[str] = None
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None

class RunFlowRequest(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

@app.post("/api/run-flow")
async def run_flow(request: RunFlowRequest):
    try:
        logs = []
        final_output = "No LLM node found or executed"

        logs.append(f"Received {len(request.nodes)} nodes and {len(request.edges)} edges")

        for node in request.nodes:
            node_type = node.type
            node_id = node.id
            data = node.data

            logs.append(f"Processing {node_type} ({node_id})")

            if node_type == "llm":
                try:
                    model = data.model or "llama-3.1-8b-instant"
                    prompt = data.userPrompt or "Hello from React Flow!"
                    system_prompt = data.systemPrompt or "You are a helpful assistant."

                    logs.append(f"→ Calling Groq: {model}")

                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=data.temperature if data.temperature is not None else 0.7,
                        max_tokens=500,
                    )

                    content = response.choices[0].message.content.strip()
                    final_output = content
                    logs.append(f"→ Groq response: {content[:150]}...")

                except Exception as e:
                    error_msg = f"Groq error: {str(e)}"
                    logs.append(error_msg)
                    final_output = error_msg

            elif node_type == "text":
                logs.append(f"→ Text content: {data.text[:80] if data.text else '(empty)'}...")
            elif node_type == "slack":
                logs.append(f"→ Slack would send to {data.channel}: {data.messagePreview}")

        return {
            "success": True,
            "output": final_output,
            "logs": logs,
            "node_count": len(request.nodes),
            "edge_count": len(request.edges),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/health")
async def health_check():
    key_loaded = bool(os.getenv("GROQ_API_KEY"))
    return {
        "status": "ok",
        "groq_key_loaded": key_loaded,
        "message": "Backend is running" if key_loaded else "Warning: GROQ_API_KEY missing"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )