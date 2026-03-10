#!/usr/bin/env python3
"""
Start script for Render.com: reads PORT from environment and runs uvicorn.
Use this as the Start Command in Render (no shell expansion needed).
"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
