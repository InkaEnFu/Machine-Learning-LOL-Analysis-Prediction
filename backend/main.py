from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.routers.predict import router as predict_router

app = FastAPI(title="LoL Rank Predictor")

app.include_router(predict_router, prefix="/api")

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")
