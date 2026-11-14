package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;
import java.awt.image.BufferedImage;
import java.awt.Color;

/**
 * Detects high-level state from screen pixels.
 * Two modes:
 * 1. Game state (Drift/Nitrous/Crash) - pixel heuristics for racing games
 * 2. AutoML pipeline phase - OCR via Tess4J + regex classification
 *
 * STATUS: Complete (was WIP/stub previously)
 */
public class VisionStateDetector {

    // ------------------------------------------------------------------
    // Game State POJO
    // ------------------------------------------------------------------
    public static class GameState {
        public boolean isDrifting;
        public boolean isNitrous;
        public boolean isCrashed;
        public double speedEstimate; // Normalized 0-1

        @Override
        public String toString() {
            return String.format("Drift:%s Nitrous:%s Crash:%s Speed:%.2f",
                    isDrifting, isNitrous, isCrashed, speedEstimate);
        }
    }

    /** AutoML pipeline phase inferred from screen text. */
    public enum AutoMLPhase {
        UNKNOWN, UPLOADING, PREPROCESSING, FEATURE_SELECTION, TRAINING, TUNING, RESULTS
    }

    // Pixel-heuristic constants (tunable)
    private static final int NITRO_COLOR_MIN_BLUE = 150;
    private static final int NITRO_COLOR_MAX_RED = 100;

    // Previous frame brightness row for drift / motion detection
    private double[] prevFrame = null;

    // ==================================================================
    // 1. GAME STATE DETECTION (pixel heuristics)
    // ==================================================================

    /**
     * Analyzes a single frame for high-level game state signals.
     */
    public GameState analyze(BufferedImage img) {
        GameState state = new GameState();
        if (img == null)
            return state;

        int w = img.getWidth();
        int h = img.getHeight();

        // 1a. Nitrous — blue bar in bottom-centre quadrant
        long blueCount = 0;
        int pixelCount = 0;
        for (int y = h - (h / 4); y < h; y++) {
            for (int x = w / 3; x < (w * 2) / 3; x++) {
                Color c = new Color(img.getRGB(x, y));
                if (c.getBlue() > NITRO_COLOR_MIN_BLUE && c.getRed() < NITRO_COLOR_MAX_RED)
                    blueCount++;
                pixelCount++;
            }
        }
        state.isNitrous = pixelCount > 0 && ((double) blueCount / pixelCount) > 0.05;

        // 1b. Crash — dominant red flash in screen centre
        long redCount = 0;
        int centrePixels = 0;
        for (int y = h / 4; y < (h * 3) / 4; y++) {
            for (int x = w / 4; x < (w * 3) / 4; x++) {
                Color c = new Color(img.getRGB(x, y));
                if (c.getRed() > 200 && c.getGreen() < 80 && c.getBlue() < 80)
                    redCount++;
                centrePixels++;
            }
        }
        state.isCrashed = centrePixels > 0 && ((double) redCount / centrePixels) > 0.30;

        // 1c. Drift — inter-frame lateral brightness shift
        double[] currentRow = computeBrightnessRow(img, h / 2, w);
        if (prevFrame != null && currentRow.length == prevFrame.length) {
            double diff = 0;
            for (int i = 0; i < currentRow.length; i++)
                diff += Math.abs(currentRow[i] - prevFrame[i]);
            double avgDiff = diff / currentRow.length;
            state.isDrifting = avgDiff > 20.0;
            state.speedEstimate = Math.min(1.0, avgDiff / 60.0);
        }
        prevFrame = currentRow;

        return state;
    }

    private double[] computeBrightnessRow(BufferedImage img, int row, int width) {
        double[] result = new double[width];
        for (int x = 0; x < width; x++) {
            Color c = new Color(img.getRGB(x, row));
            result[x] = (c.getRed() + c.getGreen() + c.getBlue()) / 3.0;
        }
        return result;
    }

    // ==================================================================
    // 2. AUTOML PIPELINE PHASE DETECTION (OCR via Tess4J)
    // ==================================================================

    /**
     * Captures a full screenshot, runs OCR, and classifies the current
     * AutoML pipeline phase.
     */
    public AutoMLPhase detectAutoMLPhase() {
        try {
            BufferedImage screenshot = captureScreen();
            String ocrText = extractTextFromImage(screenshot);
            return classifyPhase(ocrText);
        } catch (Exception e) {
            LogService.error("VisionStateDetector", "Phase detection failed: " + e.getMessage());
            return AutoMLPhase.UNKNOWN;
        }
    }

    /**
     * Captures the primary monitor into a BufferedImage using java.awt.Robot.
     */
    public BufferedImage captureScreen() throws Exception {
        java.awt.Dimension screenSize = java.awt.Toolkit.getDefaultToolkit().getScreenSize();
        java.awt.Rectangle screenRect = new java.awt.Rectangle(screenSize);
        return new java.awt.Robot().createScreenCapture(screenRect);
    }

    /**
     * Runs Tess4J OCR on a BufferedImage.
     * Tessdata is loaded from the user-writable path populated by the setup script:
     * %USERPROFILE%\AppData\Local\Tesseract-OCR\tessdata\eng.traineddata
     */
    public String extractTextFromImage(BufferedImage image) {
        try {
            net.sourceforge.tess4j.Tesseract tesseract = new net.sourceforge.tess4j.Tesseract();

            String tessDataPath = System.getProperty("user.home")
                    + java.io.File.separator + "AppData"
                    + java.io.File.separator + "Local"
                    + java.io.File.separator + "Tesseract-OCR"
                    + java.io.File.separator + "tessdata";

            tesseract.setDatapath(tessDataPath);
            tesseract.setLanguage("eng");
            tesseract.setPageSegMode(3); // Fully automatic page segmentation
            tesseract.setOcrEngineMode(1); // LSTM neural net for best accuracy

            return tesseract.doOCR(image);

        } catch (Exception e) {
            LogService.warn("VisionStateDetector",
                    "OCR unavailable (Tess4J not available or tessdata missing): " + e.getMessage());
            return "";
        }
    }

    /**
     * Maps OCR text to the current AutoML pipeline phase using keyword matching.
     */
    public AutoMLPhase classifyPhase(String text) {
        if (text == null || text.isBlank())
            return AutoMLPhase.UNKNOWN;
        String lower = text.toLowerCase();

        if (lower.contains("upload") || lower.contains("drag") || lower.contains("select file"))
            return AutoMLPhase.UPLOADING;
        if (lower.contains("preprocess") || lower.contains("normali") || lower.contains("scaling")
                || lower.contains("imputation"))
            return AutoMLPhase.PREPROCESSING;
        if (lower.contains("feature selection") || lower.contains("igpso") || lower.contains("genetic")
                || lower.contains("pso") || lower.contains("iteration"))
            return AutoMLPhase.FEATURE_SELECTION;
        if (lower.contains("training") || lower.contains("classifier") || lower.contains("random forest")
                || lower.contains("svm") || lower.contains("xgboost"))
            return AutoMLPhase.TRAINING;
        if (lower.contains("tuning") || lower.contains("hyperparameter") || lower.contains("grid search"))
            return AutoMLPhase.TUNING;
        if (lower.contains("result") || lower.contains("accuracy") || lower.contains("confusion matrix")
                || lower.contains("f1") || lower.contains("precision"))
            return AutoMLPhase.RESULTS;

        return AutoMLPhase.UNKNOWN;
    }
}
