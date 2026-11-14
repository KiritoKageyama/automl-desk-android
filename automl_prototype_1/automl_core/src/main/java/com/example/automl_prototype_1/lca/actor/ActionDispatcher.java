package com.example.automl_prototype_1.lca.actor;

import com.example.automl_prototype_1.service.LogService;
import java.awt.Robot;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.util.List;
import java.util.Map;

/**
 * The "Hands" of the Agent.
 * Executes actions received from the Brain.
 */
public class ActionDispatcher {

    private static ActionDispatcher instance;
    private final Robot robot;

    private ActionDispatcher() {
        try {
            this.robot = new Robot();
            this.robot.setAutoDelay(10); // Small delay for realism
        } catch (Exception e) {
            throw new RuntimeException("Failed to init Robot: " + e.getMessage());
        }
    }

    public static synchronized ActionDispatcher getInstance() {
        if (instance == null) {
            instance = new ActionDispatcher();
        }
        return instance;
    }

    /**
     * Executes an action defined by the Agent's JSON response.
     */
    public void dispatch(Map<String, Object> actionPlan) {
        if (actionPlan == null || actionPlan.isEmpty())
            return;

        String actionType = (String) actionPlan.get("action");
        if ("WAIT".equals(actionType))
            return;

        if (!com.example.automl_prototype_1.service.DesktopAutomationGate.getInstance()
                .requireEnabled("ActionDispatcher")) {
            return;
        }

        LogService.info("LCA-Hand", "Executing: " + actionType);

        try {
            if ("CLICK".equals(actionType) || "CLICK_CLOSE".equals(actionType) || "CLICK_MIN".equals(actionType)) {
                // Agent returns target_node as bbox [x, y, w, h] (JSON numbers)
                int[] bbox = extractBBox(actionPlan.get("target_node"));
                if (bbox != null) {
                    int cx = bbox[0] + (bbox[2] / 2);
                    int cy = bbox[1] + (bbox[3] / 2);

                    robot.mouseMove(cx, cy);
                    robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
                    robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
                }
            } else if ("KEY_TYPE".equals(actionType)) {
                Object textObj = actionPlan.get("text");
                if (textObj == null) {
                    textObj = actionPlan.get("keys");
                }
                if (textObj != null) {
                    typeText(String.valueOf(textObj));
                }
            }
        } catch (Exception e) {
            LogService.error("LCA-Hand", "Action failed: " + e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private int[] extractBBox(Object targetNodeObj) {
        if (targetNodeObj instanceof List) {
            List<?> bbox = (List<?>) targetNodeObj;
            if (bbox.size() < 4)
                return null;
            return new int[] {
                    ((Number) bbox.get(0)).intValue(),
                    ((Number) bbox.get(1)).intValue(),
                    ((Number) bbox.get(2)).intValue(),
                    ((Number) bbox.get(3)).intValue()
            };
        }
        if (targetNodeObj instanceof int[]) {
            int[] arr = (int[]) targetNodeObj;
            return arr.length >= 4 ? arr : null;
        }
        if (targetNodeObj instanceof Map) {
            Object nested = ((Map<String, Object>) targetNodeObj).get("bbox");
            return extractBBox(nested);
        }
        return null;
    }

    private void typeText(String text) {
        for (char c : text.toCharArray()) {
            int keyCode = KeyEvent.getExtendedKeyCodeForChar(c);
            if (keyCode == KeyEvent.VK_UNDEFINED) {
                continue;
            }
            boolean upper = Character.isUpperCase(c);
            if (upper) {
                robot.keyPress(KeyEvent.VK_SHIFT);
            }
            robot.keyPress(keyCode);
            robot.keyRelease(keyCode);
            if (upper) {
                robot.keyRelease(KeyEvent.VK_SHIFT);
            }
        }
    }
}
