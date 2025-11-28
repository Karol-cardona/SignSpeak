package com.example.SignSpeakBackend.controller;

import com.example.SignSpeakBackend.model.FrameData;
import com.example.SignSpeakBackend.service.FrameBufferService;
import com.example.SignSpeakBackend.service.MLSystemService;
import com.example.SignSpeakBackend.service.SimulatedTranslationService; // <--- Importante
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
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
    private final SimulatedTranslationService simulatedTranslationService; // <--- Aggiunto
    private final SimpMessagingTemplate template;

    // --- SWITCH DI CONFIGURAZIONE ---
    // Imposta su FALSE per usare il vero ML Python.
    // Imposta su TRUE per testare solo la connessione Java-React.
    private final boolean USE_SIMULATION = true;

    public WebSocketController(FrameBufferService frameBufferService,
                               MLSystemService mlSystemService,
                               SimulatedTranslationService simulatedTranslationService, // <--- Iniettato
                               SimpMessagingTemplate template) {
        this.frameBufferService = frameBufferService;
        this.mlSystemService = mlSystemService;
        this.simulatedTranslationService = simulatedTranslationService; // <--- Assegnato
        this.template = template;
    }

    @MessageMapping("/frame")
    public void receiveFrame(FrameData frameData) {
        frameData.setReceivedAt(System.currentTimeMillis());
        frameData.setSequenceNumber(frameBufferService.getNextSequenceNumber());

        String meetingId = "default";
        if (frameData.getUserInfo() != null) {
            meetingId = frameData.getUserInfo().getMeetingId();
        }

        frameBufferService.addFrame(frameData);

        // Se il buffer è pieno...
        if (frameBufferService.getBufferSize() >= frameBufferService.getChunkThreshold()) {

            List<FrameData> chunk = frameBufferService.getAndClearBuffer();

            if (USE_SIMULATION) {
                // --- MODO TEST: Chiama il servizio simulato ---
                logger.info("Buffer full. Sending SIMULATED response to {}", meetingId);
                simulatedTranslationService.sendSimulatedTranslation(meetingId);
            } else {
                // --- MODO REALE: Chiama il servizio ML Python ---
                mlSystemService.processFramesAndBroadcast(meetingId, chunk);
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
}