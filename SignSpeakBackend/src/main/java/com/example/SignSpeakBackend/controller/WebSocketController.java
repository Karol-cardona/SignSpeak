package com.example.SignSpeakBackend.controller;

import com.example.SignSpeakBackend.model.FrameData;
import com.example.SignSpeakBackend.service.FrameBufferService;
import com.example.SignSpeakBackend.service.MLSystemService;
import com.example.SignSpeakBackend.service.SimulatedTranslationService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Controller;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Controller
public class WebSocketController {

    private static final Logger logger = LoggerFactory.getLogger(WebSocketController.class);

    private final FrameBufferService frameBufferService;
    private final MLSystemService mlSystemService;
    private final SimulatedTranslationService simulatedTranslationService;
    private final SimpMessagingTemplate template;

    private final boolean USE_SIMULATION = false;

    public WebSocketController(FrameBufferService frameBufferService,
                               MLSystemService mlSystemService,
                               SimulatedTranslationService simulatedTranslationService,
                               SimpMessagingTemplate template) {
        this.frameBufferService = frameBufferService;
        this.mlSystemService = mlSystemService;
        this.simulatedTranslationService = simulatedTranslationService;
        this.template = template;
    }

    @MessageMapping("/frame")
    public void handleFrame(@Payload FrameData frameData) {
        // 1. Assegna timestamp server-side
        frameData.setReceivedAt(System.currentTimeMillis());
        frameData.setSequenceNumber(frameBufferService.getNextSequenceNumber());

        // 2. Aggiungi al buffer e controlla SE è pieno in un colpo solo
        List<FrameData> chunk = frameBufferService.addFrameAndGetChunkIfReady(frameData);

        // 3. Se abbiamo un chunk pronto (es. 3 frame), lo spediamo SUBITO al Python
        if (chunk != null && !chunk.isEmpty()) {
            if (USE_SIMULATION) {
                // Logica simulazione...
            } else {
                // Invio sincrono ottimizzato (Python ora è veloce)
                // Nota: Non usiamo @Async qui per garantire l'ordine dei pacchetti
                mlSystemService.sendFramesToML(chunk);
            }
        }
    }

    @MessageMapping("/speak")
    public void broadcastAudio(@Payload Map<String, String> payload) {
        try {
            String meetingId = payload.get("meetingId");
            String textToSpeak = payload.get("text");

            if (meetingId != null && textToSpeak != null) {
                Map<String, String> response = new HashMap<>();
                response.put("type", "AUDIO_COMMAND");
                response.put("text", textToSpeak);
                // Aggiungiamo anche meetingId per sicurezza nel filtro frontend
                response.put("meetingId", meetingId);

                String destination = "/topic/meeting/" + meetingId;
                template.convertAndSend(destination, response);
                logger.info("Audio command sent to: {}", destination);
            }
        } catch (Exception e) {
            logger.error("Error in broadcastAudio", e);
        }
    }


    @MessageMapping("/clear")
    public void clearSession(@Payload Map<String, String> payload) {
        String meetingId = payload.get("meetingId");
        logger.info("Received CLEAR command for meeting: {}", meetingId);

        if (USE_SIMULATION) {
            simulatedTranslationService.stopSimulation(meetingId);
        }

        frameBufferService.clearBuffer();

        if (!USE_SIMULATION) {
            mlSystemService.resetMLContext();
        }

        if (meetingId != null && !meetingId.isEmpty()) {
            Map<String, String> response = new HashMap<>();
            response.put("type", "CLEAR");
            response.put("meetingId", meetingId);

            String destination = "/topic/meeting/" + meetingId;
            template.convertAndSend(destination, response);
            logger.info("Broadcasted CLEAR to {}", destination);
        }
    }
}