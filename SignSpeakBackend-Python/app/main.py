import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import numpy as np

# --- IMPORT DELLA TUA LOGICA ML ESISTENTE ---
from app.dependencies import get_model
from app.session_state import UserSession
from app.preprocessing import transform_frame_to_numpy
from app.llm_handler import AdvancedSentenceCorrector

# Configurazione Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- 1. Definizione Modelli Dati (Interfaccia con Java) ---
# Questi modelli rispecchiano ESATTAMENTE ciò che invia il backend Java

class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: float

class HandInfo(BaseModel):
    score: float
    index: int
    categoryName: str
    displayName: str

class UserInfo(BaseModel):
    meetingId: Optional[str] = "default"
    userStatus: Optional[str] = None

class FrameData(BaseModel):
    timestamp: float
    receivedAt: Optional[int] = None
    sequenceNumber: Optional[int] = None
    landmarks: List[List[Landmark]] = []
    handedness: List[List[HandInfo]] = []
    userInfo: Optional[UserInfo] = None

class MLResult(BaseModel):
    prediction: Optional[str] = None
    status: str # "word_added" | "end_of_sentence" | "processing" | "partial"
    current_words: List[str] = Field(default_factory=list)
    sentence: Optional[str] = None
    detail: Optional[str] = None

class MLResponse(BaseModel):
    results: List[MLResult]

# --- 2. Inizializzazione Globale ML ---
logger.info("Loading PyTorch Model and Gemini Handler...")
model, device = get_model() # Caricato da dependencies.py
llm_handler = AdvancedSentenceCorrector() # Caricato da llm_handler.py

# Dizionario per gestire sessioni multiple: meetingId -> UserSession
active_sessions: Dict[str, UserSession] = {}

# --- 3. Endpoints ---

@app.post("/api/predict_landmarks", response_model=MLResponse)
async def predict_landmarks(request: List[FrameData]):
    """
    Riceve un batch di frame (es. 3 frame).
    Li processa SEQUENZIALMENTE per mantenere la fluidità temporale.
    """
    if not request:
        return MLResponse(results=[])

    # Gestione Sessione
    first_frame = request[0]
    meeting_id = "default"
    if first_frame.userInfo and first_frame.userInfo.meetingId:
        meeting_id = first_frame.userInfo.meetingId

    if meeting_id not in active_sessions:
        active_sessions[meeting_id] = UserSession()

    session = active_sessions[meeting_id]
    results = []

    # --- SIMULAZIONE REALTIME ---
    # Iteriamo su ogni frame del pacchetto come se fosse un loop video
    for frame in request:
        # Trasforma
        frame_numpy = transform_frame_to_numpy(frame)

        # Processa (Normalizza -> Buffer -> Predici -> Vota)
        event = session.process_frame(frame_numpy, model, device)

        # Se succede qualcosa in questo millisecondo, lo salviamo
        if event:
            ml_result = MLResult(
                prediction=event["prediction"],
                status=event["status"],
                current_words=event.get("current_words", []),
                sentence=event.get("sentence", ""),
                detail="Realtime detection"
            )
            results.append(ml_result)

    return MLResponse(results=results)


@app.post("/api/reset_buffer")
async def reset_buffer(request: UserInfo = None): # Accettiamo opzionalmente UserInfo per resettare specifici meeting
    """
    Resetta la sessione. Se viene passato un meetingId, resetta solo quello.
    Altrimenti resetta tutto (o gestisci come preferisci).
    Nota: Il Java attuale chiama questo endpoint senza body o con json vuoto nel clearSession,
    ma possiamo adattarlo.
    """
    # Se il Java invia un reset generico, potremmo dover resettare tutte le sessioni o una default.
    # Per ora svuotiamo tutto il dizionario per sicurezza se non c'è ID specifico,
    # ma idealmente il Java dovrebbe inviare il meetingId anche qui.

    global active_sessions
    active_sessions.clear()
    logger.info("All sessions cleared via reset_buffer command")

    return {"message": "All buffers cleared successfully"}

# Avvio server (se lanciato direttamente per debug locale)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)