# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# import your jupyter code converted into agent.py
# agent.py must expose `test_pipeline(query: str) -> dict` which returns a Vega-Lite spec (JSON-serializable)
import agent


app = FastAPI(title="TalkToADatabase Prototype")


# allow local frontend (adjust origin if needed)
app.add_middleware(
CORSMiddleware,
allow_origins=["http://localhost:3000"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)


class QueryIn(BaseModel):
    query: str

@app.post("/query")
async def run_query(payload: QueryIn):
    q = payload.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")
    try:
    # test_pipeline should internally use the loaded CSV (agent.py) and return Vega-Lite spec
        spec = agent.test_pipeline(q)
    # spec must be JSON serialized (dict)
        return {"status": "ok", "spec": spec}
    except Exception as e:
    # surface useful error for debugging in prototype
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")