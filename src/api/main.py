from fastapi import FastAPI

from src.api.routes import router
from src.bundle import DISCLAIMER

app = FastAPI(title="Court Case Forecasting API", version="0.1.0",
              description="Duration forecasts only. " + DISCLAIMER)
app.include_router(router)
