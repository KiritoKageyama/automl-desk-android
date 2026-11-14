package com.example.automl_prototype_1.lca.perception;

import com.example.automl_prototype_1.service.LogService;
import net.sourceforge.tess4j.Tesseract;

import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;

/**
 * The "Eyes" of the Agent.
 * Captures the screen and extracts text/visual nodes.
 */
public class PerceptionService {

    private static PerceptionService instance;
    private final Robot robot;
    private final Tesseract tesseract;
    private final Rectangle screenRect;

    private PerceptionService() {
        try {
            this.robot = new Robot();
            this.screenRect = new Rectangle(Toolkit.getDefaultToolkit().getScreenSize());

            this.tesseract = new Tesseract();
            // Assuming tessdata is in project root or standard location
            // We might need to configure this path dynamically
            this.tesseract.setDatapath("tessdata");
            this.tesseract.setLanguage("eng");

            LogService.info("LCA-Eye", "PerceptionService Initialized. Screen Size: " + screenRect);
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize PerceptionService: " + e.getMessage());
        }
    }

    public static synchronized PerceptionService getInstance() {
        if (instance == null) {
            instance = new PerceptionService();
        }
        return instance;
    }

    /**
     * Captures the current state of the screen and constructs a "UI Snapshot".
     * 
     * @return A Map representing the UI Graph (JSON-serializable).
     */
    public Map<String, Object> captureSnapshot() {
        Map<String, Object> snapshot = new HashMap<>();
        long start = System.currentTimeMillis();

        try {
            // 1. Capture Screen
            BufferedImage screenshot = robot.createScreenCapture(screenRect);

            // 2. OCR (Simulating Accessibility Tree extraction)
            // For a real high-speed loop, we might skip full OCR every frame
            // or use a faster method. For prototype, we do it.
            List<Map<String, Object>> nodes = new ArrayList<>();

            // OCR is slow. We might assume the Agent only needs a Screenshot for the CNN
            // and we OCR only on demand?
            // BUT, the 'FastActor' loop iterates over "nodes".
            // So we need at least some nodes.
            // Let's implement a "Fast Scan" -> Just the image for CNN,
            // The Python side also has a 'ui_elements =
            // extract_accessibility_tree(screen_pixels)'
            // which implies Python might do it, OR we do it.
            // Since Tesseract is Java/JNA, we do it here.

            // Note: Tesseract 'getWords' gives bounding boxes.
            try {
                // This usually returns a list of Words/Rectangles
                List<net.sourceforge.tess4j.Word> words = tesseract.getWords(screenshot,
                        net.sourceforge.tess4j.ITessAPI.TessPageIteratorLevel.RIL_WORD);
                for (net.sourceforge.tess4j.Word w : words) {
                    Map<String, Object> node = new HashMap<>();
                    node.put("text", w.getText());
                    Rectangle r = w.getBoundingBox();
                    node.put("bbox", new int[] { r.x, r.y, r.width, r.height });
                    node.put("confidence", w.getConfidence());
                    nodes.add(node);
                }
            } catch (Throwable t) {
                // If OCR fails (e.g. native lib missing), we continue with 0 nodes
                // The Agent will rely purely on Visual CNN (if implemented to scan grid)
                // or fail gracefully.
                LogService.warn("LCA-Eye", "OCR extraction failed (continuing with visual-only): " + t.getMessage());
            }

            // 3. Compress Image for Transport (Base64)
            // Python needs the raw image for CNN embeddings
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(screenshot, "jpg", baos); // JPG for speed/size
            String base64Img = Base64.getEncoder().encodeToString(baos.toByteArray());

            snapshot.put("screenshot_base64", base64Img);
            snapshot.put("nodes", nodes);
            snapshot.put("timestamp", System.currentTimeMillis());
            snapshot.put("width", screenRect.width);
            snapshot.put("height", screenRect.height);

            // Context Inference (Placeholder)
            Map<String, String> context = new HashMap<>();
            context.put("id", "unknown_app"); // Needs OS hooks (JNA) to get Active Window Title
            context.put("domain", "general");
            snapshot.put("context", context);

            long duration = System.currentTimeMillis() - start;
            if (duration > 500) {
                LogService.warn("LCA-Eye", "Snapshot took " + duration + "ms (Slow!).");
            }

        } catch (Exception e) {
            LogService.error("LCA-Eye", "Capture failed: " + e.getMessage());
        }

        return snapshot;
    }
}
