import uvicorn
import sys
import asyncio

if __name__ == "__main__":
    # 1. Force the compatible Event Loop for Windows + Postgres
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 2. Run the Server programmatically
    print("🚀 Launching Server with Windows Fix...")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)