from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

# This creates the "app" variable that uvicorn looks for
app = FastAPI(title=settings.PROJECT_NAME)

# Include the routes from endpoints.py
app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)