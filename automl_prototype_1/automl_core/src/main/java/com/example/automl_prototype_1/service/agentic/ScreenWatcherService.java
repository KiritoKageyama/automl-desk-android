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
    /**
     * Captures the screen and returns the raw resized BufferedImage (Color).
     * Useful for Vision State Detection.
     */
    public BufferedImage captureImage() {
        return captureImage(targetWidth, targetHeight);
    }

    public BufferedImage captureImage(int width, int height) {
        if (robot == null)
            return null;
        BufferedImage screenshot = robot.createScreenCapture(captureArea);
        return resizeColor(screenshot, width, height);
    }

    /**
     * Captures the current screen state and returns it as a flattened double array
     * (Tensor).
     * Format: Grayscale, normalized [0, 1].
     */
    public double[] captureState() {
        BufferedImage img = captureImage();
        // processImageToTensor converts to grayscale tensor automatically
        return processImageToTensor(img);
    }

    public BufferedImage resizeColor(BufferedImage original, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g = resized.createGraphics();
        g.drawImage(original, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }

    public double[] processImageToTensor(BufferedImage img) {
        if (img == null) {
            return new double[0];
        }
        int w = img.getWidth();
        int h = img.getHeight();
        double[] tensor = new double[w * h];

        // captureImage() returns TYPE_INT_RGB — do not assume DataBufferByte / GRAY.
        // Convert to luminance via getRGB so INT_RGB and BYTE_GRAY both work.
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                tensor[y * w + x] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
            }
        }

        return tensor;
    }
}
