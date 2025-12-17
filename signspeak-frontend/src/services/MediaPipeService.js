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

                let categoryName = handedness.categoryName;
                let targetIndex = (categoryName === 'Right') ? 0 : 1;

                // --- FIX 2: COORDINATE MIRRORING ---
                const formattedPoints = rawLandmarks.map(lm => ({
                    x: 1.0 - lm.x,
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

            if (this.results) {
                const formattedData = this.formatResults(this.results);

                // --- MODIFICA QUI ---
                // Controlliamo se ci sono dati nella mano destra (index 0) o sinistra (index 1)
                const hasRightHand = formattedData.landmarks[0] && formattedData.landmarks[0].length > 0;
                const hasLeftHand = formattedData.landmarks[1] && formattedData.landmarks[1].length > 0;

                // Inviamo i dati SOLO se almeno una mano è rilevata e valida (non filtrata)
                if (hasRightHand || hasLeftHand) {
                    this.onResultsCallback(formattedData, nowInMs);
                }
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