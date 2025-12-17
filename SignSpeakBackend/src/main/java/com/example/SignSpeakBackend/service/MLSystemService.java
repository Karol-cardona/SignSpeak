package com.example.SignSpeakBackend.service;

import com.example.SignSpeakBackend.model.FrameData;
import com.example.SignSpeakBackend.model.MLResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class MLSystemService {

    private static final Logger logger = LoggerFactory.getLogger(MLSystemService.class);

    @Value("${ml.system.url:http://localhost:8000/api/predict_landmarks}")
    private String mlSystemUrl;

    private final RestTemplate restTemplate;
    private final SimpMessagingTemplate messagingTemplate;

    public MLSystemService(RestTemplate restTemplate, SimpMessagingTemplate messagingTemplate) {
        this.restTemplate = restTemplate;
        this.messagingTemplate = messagingTemplate;
    }

    /**
     * Invia un chunk di frame al ML e gestisce la risposta.
     * Metodo rinominato per combaciare con la chiamata del Controller.
     */
    public void sendFramesToML(List<FrameData> frames) {
        if (frames == null || frames.isEmpty()) return;

        // Recuperiamo il meetingId dal primo frame del chunk per sapere dove mandare la risposta
        String meetingId = "default";
        if (frames.get(0).getUserInfo() != null && frames.get(0).getUserInfo().getMeetingId() != null) {
            meetingId = frames.get(0).getUserInfo().getMeetingId();
        }

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<List<FrameData>> request = new HttpEntity<>(frames, headers);

            // Chiamata POST all'API Python
            MLResponse response = restTemplate.postForObject(mlSystemUrl, request, MLResponse.class);

            if (response != null && response.getResults() != null) {
                for (MLResponse.MLResult result : response.getResults()) {
                    broadcastResult(meetingId, result);
                }
            }

        } catch (Exception e) {
            logger.error("Error communicating with ML system: {}", e.getMessage());
        }
    }

    private void broadcastResult(String meetingId, MLResponse.MLResult result) {
        String destination = "/topic/meeting/" + meetingId;
        Map<String, Object> message = new HashMap<>();

        if ("word_added".equals(result.getStatus())) {
            message.put("type", "PARTIAL");
            message.put("words", result.getCurrentWords());
            message.put("last_prediction", result.getPrediction());

            // Log ridotto per velocità
            logger.info(">> New Word: {}", result.getPrediction());

        } else if ("end_of_sentence".equals(result.getStatus())) {
            message.put("type", "FINAL");
            message.put("text", result.getSentence());

            logger.info(">> Sentence: {}", result.getSentence());
        }

        if (!message.isEmpty()) {
            messagingTemplate.convertAndSend(destination, message);
        }
    }

    public void resetMLContext() {
        try {
            String url = mlSystemUrl.replace("/predict_landmarks", "/reset_buffer");
            restTemplate.postForObject(url, null, String.class);
            logger.info("Sent RESET command to ML Service");
        } catch (Exception e) {
            logger.error("Failed to reset ML Service: {}", e.getMessage());
        }
    }
}