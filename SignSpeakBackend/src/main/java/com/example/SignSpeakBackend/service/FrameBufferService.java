package com.example.SignSpeakBackend.service;

import com.example.SignSpeakBackend.model.FrameData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

//@Service
//public class FrameBufferService {
//
//    private static final Logger logger = LoggerFactory.getLogger(FrameBufferService.class);
//
//    private final ConcurrentLinkedQueue<FrameData> buffer = new ConcurrentLinkedQueue<>();
//    private final MLSystemService mlSystemService;
//    private final AtomicInteger sequenceCounter = new AtomicInteger(0);
//
//    @Value("${frame.selection.count:30}")
//    private int frameSelectionCount;
//
//    public FrameBufferService(MLSystemService mlSystemService) {
//        this.mlSystemService = mlSystemService;
//    }
//
//    public int getNextSequenceNumber() {
//        return sequenceCounter.getAndIncrement();
//    }
//
//    public void addFrame(FrameData frameData) {
//        buffer.offer(frameData);
//        logger.debug("Frame added to buffer. Buffer size: {}", buffer.size());
//    }
//
//    @Scheduled(fixedRate = 5000) // Every 5 seconds
//    public void processBuffer() {
//        if (buffer.isEmpty()) {
//            logger.info("Buffer is empty, skipping processing");
//            return;
//        }
//
//        List<FrameData> allFrames = new ArrayList<>(buffer);
//        List<FrameData> selectedFrames = selectDistributedFrames(allFrames, frameSelectionCount);
//
//        logger.info("Processing buffer. Total frames: {}, Selected frames: {}",
//                allFrames.size(), selectedFrames.size());
//
//        // Send to ML system
//        mlSystemService.sendFrames(selectedFrames);
//
//        // Clear the buffer
//        buffer.clear();
//    }
//
//    /**
//     * Selects frames that are evenly distributed across the time window
//     */
//    private List<FrameData> selectDistributedFrames(List<FrameData> frames, int count) {
//        if (frames.isEmpty()) {
//            return new ArrayList<>();
//        }
//
//        if (frames.size() <= count) {
//            return new ArrayList<>(frames);
//        }
//
//        List<FrameData> selected = new ArrayList<>();
//        double step = (double) frames.size() / count;
//
//        for (int i = 0; i < count; i++) {
//            int index = (int) Math.round(i * step);
//            if (index >= frames.size()) {
//                index = frames.size() - 1;
//            }
//            selected.add(frames.get(index));
//        }
//
//        return selected;
//    }
//
//    public int getBufferSize() {
//        return buffer.size();
//    }
//
//    public void setFrameSelectionCount(int count) {
//        this.frameSelectionCount = count;
//        logger.info("Frame selection count updated to: {}", count);
//    }
//
//    public int getFrameSelectionCount() {
//        return frameSelectionCount;
//    }
//}

@Service
public class FrameBufferService {

    private static final Logger logger = LoggerFactory.getLogger(FrameBufferService.class);

    private final ConcurrentLinkedQueue<FrameData> buffer = new ConcurrentLinkedQueue<>();
    private final MLSystemService mlSystemService;
    private final AtomicInteger sequenceCounter = new AtomicInteger(0);

    @Value("${buffer.max.size:120}")
    private int bufferMaxSize;

    @Value("${buffer.evacuate.count:80}")
    private int evacuateCount;

    @Value("${buffer.retain.count:40}")
    private int retainCount;

    public FrameBufferService(MLSystemService mlSystemService) {
        this.mlSystemService = mlSystemService;
    }

    public int getNextSequenceNumber() {
        return sequenceCounter.getAndIncrement();
    }

    public void addFrame(FrameData frameData) {
        buffer.offer(frameData);
        int currentSize = buffer.size();
        logger.debug("Frame added to buffer. Buffer size: {}", currentSize);

        // Check if we've reached the threshold for evacuation
        if (currentSize >= bufferMaxSize) {
            logger.info("Buffer reached {} frames, triggering evacuation", currentSize);
            evacuateFrames();
        }
    }
    /**
     * Evacuates the oldest frames to ML system and keeps the newest frames in buffer
     * Called automatically when buffer reaches the configured threshold (default: 120 frames)
     */
    private synchronized void evacuateFrames() {
        int currentSize = buffer.size();

        if (currentSize < bufferMaxSize) {
            logger.debug("Buffer size {} below threshold {}, skipping evacuation",
                    currentSize, bufferMaxSize);
            return;
        }

        logger.info("Starting evacuation. Buffer size: {}", currentSize);

        // Extract frames in order (FIFO)
        List<FrameData> allFrames = new ArrayList<>();
        FrameData frame;
        while ((frame = buffer.poll()) != null) {
            allFrames.add(frame);
        }

        // Validate we have enough frames
        if (allFrames.size() < evacuateCount) {
            logger.warn("Not enough frames to evacuate. Expected at least {}, got {}",
                    evacuateCount, allFrames.size());
            // Put all frames back
            allFrames.forEach(buffer::offer);
            return;
        }

        // Split: oldest 80 frames to evacuate, newest 40 to retain
        List<FrameData> toEvacuate = allFrames.subList(0, evacuateCount);
        List<FrameData> toRetain = allFrames.subList(evacuateCount, allFrames.size());

        logger.info("Evacuating {} frames (seq #{} to #{}), retaining {} frames (seq #{} to #{})",
                toEvacuate.size(),
                toEvacuate.get(0).getSequenceNumber(),
                toEvacuate.get(toEvacuate.size() - 1).getSequenceNumber(),
                toRetain.size(),
                toRetain.isEmpty() ? "N/A" : toRetain.get(0).getSequenceNumber(),
                toRetain.isEmpty() ? "N/A" : toRetain.get(toRetain.size() - 1).getSequenceNumber());

        // Send evacuated frames to ML system
        mlSystemService.sendFrames(new ArrayList<>(toEvacuate));

        // Put retained frames back in buffer (maintaining order)
        toRetain.forEach(buffer::offer);

        logger.info("Evacuation complete. Buffer size now: {}", buffer.size());
    }

    public int getBufferSize() {
        return buffer.size();
    }

    public int getBufferMaxSize() {
        return bufferMaxSize;
    }

    public void setBufferMaxSize(int maxSize) {
        this.bufferMaxSize = maxSize;
        logger.info("Buffer max size updated to: {}", maxSize);
    }

    public int getEvacuateCount() {
        return evacuateCount;
    }

    public void setEvacuateCount(int count) {
        this.evacuateCount = count;
        logger.info("Evacuate count updated to: {}", count);
    }

    public int getRetainCount() {
        return retainCount;
    }

    public void setRetainCount(int count) {
        this.retainCount = count;
        logger.info("Retain count updated to: {}", count);
    }
}