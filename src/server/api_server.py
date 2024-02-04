from fastapi import FastAPI
from src.engine.datasets import MixteraDataset
from src.engine.operators import Query
from fastapi.responses import StreamingResponse
from src.server.protocol import ReadDatasetRequest

app = FastAPI()
dataset = None

@app.get("/keys")
async def read_keys():
    return dataset.read_keys()
            
@app.get("/key/{key_name}")
async def read_key_values(key_name: str):
    return dataset.find_by_key(key_name)

@app.post("/data")
async def read_file(read_request: ReadDatasetRequest):
    if read_request.streaming:
        return StreamingResponse(dataset.stream_values(keys = read_request.fids))
    else:
        res = dataset.read_values(keys=read_request.fids)
        return res
    
if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="Mixtera API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    args = parser.parse_args()

    dataset = MixteraDataset.from_folder(args.dataset)
    uvicorn.run(app, host=args.host, port=args.port)