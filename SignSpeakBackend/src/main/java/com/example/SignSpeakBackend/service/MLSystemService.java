package com.example.SignSpeakBackend.service;

import com.example.SignSpeakBackend.model.FrameData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.messaging.MessagingException;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;

@Service
public class MLSystemService {

    private static final Logger logger = LoggerFactory.getLogger(MLSystemService.class);

    @Value("${ml.system.url:http://host.docker.internal:8000/api/predict_landmarks}")
    private String mlSystemUrl;

    private final NotificationService notificationService;

    private final RestTemplate restTemplate;

    public MLSystemService(NotificationService notificationService) {
        this.restTemplate = new RestTemplate();
        this.notificationService = notificationService;
    }

    public void sendFrames(List<FrameData> frames) {
        try {
            logger.info("Sending {} frames to ML system at {}", frames.size(), mlSystemUrl);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<List<FrameData>> request = new HttpEntity<>(frames, headers);

             String response = restTemplate.postForObject(mlSystemUrl, request, String.class);
             logger.info("ML system response: {}", response);

            notificationService.sendToClients(response);

            logger.info("Frames that would be sent to ML system:");
            frames.forEach(frame -> {
                String handInfo = "";
                if (frame.getHandedness() != null && !frame.getHandedness().isEmpty()
                        && !frame.getHandedness().get(0).isEmpty()) {
                    var firstHand = frame.getHandedness().get(0).get(0);
                    handInfo = String.format(" (%s hand, confidence: %.2f)",
                            firstHand.getCategoryName(),
                            firstHand.getScore());
                }
                logger.info("  Frame seq#{} at {:.2f}ms with {} hand(s){}",
                        frame.getSequenceNumber(),
                        frame.getTimestamp(),
                        frame.getHandCount(),
                        handInfo);
            });

        } catch (MessagingException e) {
            logger.error("Error sending message to the frontend", e);
        }
        catch (Exception e) {
            logger.error("Error sending frames to ML system", e);
        }
    }
}