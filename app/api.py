from fastapi import APIRouter
from typing import List

from .schemas import FrameData
from .model_logic.PipelineManager import PipelineManager

router = APIRouter()

# Singleton pipeline (model load only once)
pipeline = PipelineManager()


@router.post("/predict_landmarks")
def predict_landmarks   (frames: List[FrameData]):
    """
    Accepts a list of FrameData objects.
    Returns detected words or sentences.
    """
    try:
        responses = pipeline.process(frames)
        return {"results": responses}
    except Exception as e:
        return {"error": str(e)}
