import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

export class MediaPipeService {
    handLandmarker = null;
    animationFrameId = null;
    lastVideoTime = -1;
    results = null;
    onResultsCallback = null;

    /**
     * Initializes the HandLandmarker model.
     * @param {function} onResults - The callback function to handle landmark results.
     */
    async initialize(onResults) {
        this.onResultsCallback = onResults;

        const vision = await FilesetResolver.forVisionTasks(
            // Use the CDN to load model files
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        // Create the HandLandmarker instance
        this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                // Use the 'lite' model for better performance
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                delegate: "GPU",
            },
            runningMode: "VIDEO",
            numHands: 2,
            // Aggiungiamo confidenza minima per ridurre il rumore
            minHandDetectionConfidence: 0.7,
            minHandPresenceConfidence: 0.7,
            minTrackingConfidence: 0.7
        });

        console.log("MediaPipe HandLandmarker initialized.");
    }

    /**
     * Starts the landmark detection loop.
     * @param {HTMLVideoElement} videoElement - The <video> element with the webcam feed.
     */
    startProcessing(videoElement) {
        if (!this.handLandmarker) {
            console.error("MediaPipe service is not initialized.");
            return;
        }
        this.predictWebcam(videoElement);
    }

    /**
     * HELPER: Formatta i risultati per il Backend Python.
     * Il backend (preprocessing.py) si aspetta RIGOROSAMENTE:
     * - Index 0: Dati Mano Destra (o lista vuota)
     * - Index 1: Dati Mano Sinistra (o lista vuota)
     */
    formatResults(results) {
        let landmarksPayload = [[], []];
        let handednessPayload = [[], []];

        if (results.landmarks) {
            for (let i = 0; i < results.landmarks.length; i++) {
                const rawLandmarks = results.landmarks[i];
                const handedness = results.handedness[i][0];

                // --- FILTRO SMART (Polso + Dita) ---
                const wrist = rawLandmarks[0];      // Polso
                const indexTip = rawLandmarks[8];   // Punta Indice
                const pinkyTip = rawLandmarks[20];  // Punta Mignolo

                // Se il polso è giù (>0.9) E ANCHE le dita sono giù (>0.9)
                // Allora la mano è morta/appoggiata -> IGNORA
                if (wrist.y > 0.90 && indexTip.y > 0.90 && pinkyTip.y > 0.90) {
                    continue;
                }

                let categoryName = handedness.categoryName; // "Left" o "Right"

                // Se noti che le mani sono scambiate (il modello predice male quando usi una mano sola),
                // DECOMMENTA queste righe per invertire le etichette:
                // categoryName = (categoryName === "Right") ? "Left" : "Right";

                let targetIndex = (categoryName === 'Right') ? 0 : 1;

                // --- FIX 2: COORDINATE MIRRORING ---
                // Python usa cv2.flip(1). Qui dobbiamo simularlo matematicamente.
                // Invertiamo l'asse X:  x_new = 1.0 - x_old
                const formattedPoints = rawLandmarks.map(lm => ({
                    x: 1.0 - lm.x,  // <--- ECCO IL TRUCCO PER L'ACCURACY
                    y: lm.y,
                    z: lm.z,
                    visibility: lm.visibility ?? 1.0
                }));

                const handInfoObj = [{
                    score: handedness.score,
                    index: handedness.index,
                    categoryName: categoryName,
                    displayName: handedness.displayName
                }];

                landmarksPayload[targetIndex] = formattedPoints;
                handednessPayload[targetIndex] = handInfoObj;
            }
        }

        return {
            timestamp: performance.now(),
            userInfo: {
                meetingId: "default",
                userStatus: "active"
            },
            landmarks: landmarksPayload,
            handedness: handednessPayload
        };
    }

    /**
     * The main prediction loop, running on every animation frame.
     */
    predictWebcam = (videoElement) => {
        const videoTime = videoElement.currentTime;
        const nowInMs = performance.now();

        // Process the frame if it's new
        if (this.lastVideoTime !== videoTime) {
            this.lastVideoTime = videoTime;

            // Perform landmark detection
            this.results = this.handLandmarker.detectForVideo(videoElement, nowInMs);

            // MODIFICA: Formattiamo SEMPRE i risultati e li inviamo, anche se vuoti.
            // Questo permette al backend di sapere che non stai facendo gesti.
            if (this.results) {
                const formattedData = this.formatResults(this.results);
                this.onResultsCallback(formattedData, nowInMs);
            }
        }

        // Continue the loop
        this.animationFrameId = requestAnimationFrame(() => this.predictWebcam(videoElement));
    };

    /**
     * Stops the landmark detection loop.
     */
    stopProcessing() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        console.log("MediaPipe processing stopped.");
    }
}