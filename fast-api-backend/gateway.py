import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

app = FastAPI()
client = httpx.AsyncClient()

SERVICES = {
    "vector": "http://127.0.0.1:8001",
    "chat": "http://127.0.0.1:8002",
    "auth": "http://127.0.0.1:8003",
    "user": "http://127.0.0.1:8004",
}

@app.get('/health')
def health():
    return "p"

# ponytail: naive reverse proxy — no retries/circuit-breaking/streaming bodies.
# Fine while each service is a /health stub; revisit if request bodies get large
# or a backend needs to be resilient to the others being down.
@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(service: str, path: str, request: Request):
    base_url = SERVICES.get(service)
    if base_url is None:
        return Response(content=f"unknown service: {service}", status_code=404)

    upstream = await client.request(
        request.method,
        f"{base_url}/{path}",
        params=request.query_params,
        headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")},
        content=await request.body(),
    )
    excluded = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in excluded}
    return Response(content=upstream.content, status_code=upstream.status_code, headers=headers)
