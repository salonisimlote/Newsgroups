import json
import asyncio

class Request:
    def __init__(self, scope, receive):
        self.scope = scope
        self._receive = receive
    def json(self):
        return self._body
    async def _load(self):
        body = b""
        while True:
            msg = await self._receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        try:
            self._body = json.loads(body) if body else {}
        except:
            self._body = {}

class Response:
    def __init__(self, data, status=200):
        self.data = data
        self.status = status

class Router:
    def __init__(self):
        self._routes = {}
        self._startup = []
        self._shutdown = []

    def _add(self, method, path, fn):
        self._routes[(method.upper(), path)] = fn

    def get(self, path):
        def dec(fn): self._add("GET", path, fn); return fn
        return dec

    def post(self, path):
        def dec(fn): self._add("POST", path, fn); return fn
        return dec

    def delete(self, path):
        def dec(fn): self._add("DELETE", path, fn); return fn
        return dec

    def on_startup(self, fn):
        self._startup.append(fn); return fn

    def on_shutdown(self, fn):
        self._shutdown.append(fn); return fn

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    for fn in self._startup: await fn()
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    for fn in self._shutdown: await fn()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        elif scope["type"] == "http":
            method = scope["method"]
            path   = scope["path"]
            handler = self._routes.get((method, path))
            req = Request(scope, receive)
            await req._load()
            if handler:
                resp = await handler(req)
            else:
                resp = Response({"detail": "Not found"}, status=404)
            body = json.dumps(resp.data).encode()
            await send({"type": "http.response.start", "status": resp.status,
                        "headers": [[b"content-type", b"application/json"]]})
            await send({"type": "http.response.body", "body": body})