from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "message": "Sign Language Detection API is running!"}


def test_predict_with_hand():
    # Make sure you have a test video named 'test_video_with_hand.mp4' in the tests directory
    video_path = os.path.join(os.path.dirname(
        __file__), "test_video_with_hand.mp4")
    if not os.path.exists(video_path):
        # Create a dummy file if it doesn't exist, to avoid errors
        with open(video_path, "w") as f:
            f.write("")

    with open(video_path, "rb") as f:
        response = client.post(
            "/predict", files={"file": ("test_video_with_hand.mp4", f, "video/mp4")})

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_no_hand():
    # A video where no hand is present. You'll need to provide such a video.
    video_path = os.path.join(os.path.dirname(
        __file__), "test_video_no_hand.mp4")
    if not os.path.exists(video_path):
        with open(video_path, "w") as f:
            f.write("")

    with open(video_path, "rb") as f:
        response = client.post(
            "/predict", files={"file": ("test_video_no_hand.mp4", f, "video/mp4")})

    assert response.status_code == 200
    assert response.json() == {"prediction": "No hand detected in the video."}
