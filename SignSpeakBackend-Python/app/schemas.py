# --- START OF FILE schemas.py ---
from pydantic import BaseModel
from typing import List, Optional


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = None


class HandednessInfo(BaseModel):
    score: float
    index: int
    categoryName: str
    displayName: str


class FrameData(BaseModel):
    # We only need the raw landmarks and handedness for the model
    landmarks: List[List[LandmarkPoint]]
    handedness: List[List[HandednessInfo]]


class PredictRequest(BaseModel):
    session_id: str  # Unique ID for the user (e.g., "user_123")
    frames: List[FrameData]  # A batch of new frames collected by the app
