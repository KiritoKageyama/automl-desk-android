package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;

import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.image.BufferedImage;
import java.awt.Toolkit;

/**
 * Service to capture specific screen regions for RL observation.
 * Uses java.awt.Robot.
 */
public class ScreenWatcherService {

    private Robot robot;
    private Rectangle captureArea;
    private int targetWidth = 84; // Standard RL input size (e.g. Atari)
    private int targetHeight = 84;

    public ScreenWatcherService() {
        try {
            this.robot = new Robot();
            // Default to full screen if not specified
            this.captureArea = new Rectangle(Toolkit.getDefaultToolkit().getScreenSize());
        } catch (Exception e) {
            LogService.error("ScreenWatcher", "Failed to initialize Robot: " + e.getMessage());
        }
    }

    /**
     * Sets the screen region to watch.
     * Use a helper tool or UI to let users select this.
     */
    public void setCaptureArea(int x, int y, int width, int height) {
        this.captureArea = new Rectangle(x, y, width, height);
    }

    /**
     * Captures the current screen state and returns it as a flattened double array
     * (Tensor).
     * Format: Grayscale, normalized [0, 1].
     */
    public double[] captureState() {
        if (robot == null)
            return new double[0];

        BufferedImage screenshot = robot.createScreenCapture(captureArea);

        // Resize (Simple nearest neighbor or similar for speed)
        BufferedImage resized = resize(screenshot, targetWidth, targetHeight);

        return processImageToTensor(resized);
    }

    private BufferedImage resize(BufferedImage original, int width, int height) {
        // Create a new image of the target size (TYPE_BYTE_GRAY requires careful
        // handling if not supported by Graphics2D directly in all JDKs)
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        java.awt.Graphics2D g = resized.createGraphics();
        g.drawImage(original, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }

    private double[] processImageToTensor(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        double[] tensor = new double[w * h];

        // Access internal data buffer if possible for speed, or getRGB
        // For TYPE_BYTE_GRAY:
        byte[] pixels = ((java.awt.image.DataBufferByte) img.getRaster().getDataBuffer()).getData();

        for (int i = 0; i < pixels.length; i++) {
            // Convert byte [-128, 127] to unsigned int [0, 255] then normalize [0, 1]
            int val = pixels[i] & 0xFF;
            tensor[i] = val / 255.0;
        }

        return tensor;
    }
}
