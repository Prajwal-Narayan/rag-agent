from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.endpoints import router
from app.core.config import settings
from app.core.database import init_db, pool

# 👇 NEW: Lifespan Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("🚀 Starting up Database Connection...")
    await init_db()
    yield
    # Shutdown logic
    print("🛑 Closing Database Connection...")
    await pool.close()

# Pass lifespan to FastAPI
app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Force Selector Loop for Windows
    import sys
    import asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)