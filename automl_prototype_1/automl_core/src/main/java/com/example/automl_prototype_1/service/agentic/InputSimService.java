package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;
import java.awt.Robot;
import java.awt.event.KeyEvent;

/**
 * Service to simulate keyboard and mouse input for controlling external
 * applications.
 * Uses java.awt.Robot.
 */
public class InputSimService {

    private Robot robot;

    public InputSimService() {
        try {
            this.robot = new Robot();
            // Auto-delay to make it more human-like and reliable for games
            this.robot.setAutoDelay(50);
        } catch (Exception e) {
            LogService.error("InputSim", "Failed to initialize Robot: " + e.getMessage());
        }
    }

    private String keyProfile = "WASD"; // Default

    public void setKeyProfile(String profile) {
        this.keyProfile = profile;
        LogService.info("InputSim", "Key Profile set to: " + profile);
    }

    /**
     * Maps an RL discrete action to a specific key press.
     * Use this for predefined discrete action spaces.
     */
    public void performAction(int action) {
        if (robot == null)
            return;
        // Legacy/Default mapping
        if (keyProfile.equals("ARROWS")) {
            // Racing Games (NFS often uses Arrows)
            switch (action) {
                case 1:
                    tapKey(KeyEvent.VK_UP);
                    break; // Accelerate
                case 2:
                    tapKey(KeyEvent.VK_LEFT);
                    break; // Left
                case 3:
                    tapKey(KeyEvent.VK_RIGHT);
                    break; // Right
                case 4:
                    tapKey(KeyEvent.VK_DOWN);
                    break; // Brake/Reverse
                case 5:
                    tapKey(KeyEvent.VK_SPACE);
                    break; // Handbrake/Nitro
                default:
                    break;
            }
        } else if (keyProfile.equals("TREX")) {
            // T-Rex (Space to Jump, Down to Duck)
            switch (action) {
                case 1:
                    tapKey(KeyEvent.VK_SPACE);
                    break; // Jump
                case 2:
                    tapKey(KeyEvent.VK_DOWN);
                    break; // Duck
                default:
                    break; // Ignore others
            }
        } else {
            // Default WASD
            switch (action) {
                case 1:
                    tapKey(KeyEvent.VK_W);
                    break;
                case 2:
                    tapKey(KeyEvent.VK_A);
                    break;
                case 3:
                    tapKey(KeyEvent.VK_D);
                    break;
                case 4:
                    tapKey(KeyEvent.VK_S);
                    break;
                case 5:
                    tapKey(KeyEvent.VK_SPACE);
                    break;
                default:
                    break;
            }
        }
    }

    /**
     * Press a key by its standard name (e.g., "W", "Space", "Enter").
     * Useful for LLM-driven actions.
     */
    public void pressKeyByName(String keyName) {
        if (robot == null || keyName == null)
            return;

        int keyCode = getKeyCode(keyName);
        if (keyCode != -1) {
            tapKey(keyCode);
        } else {
            LogService.warn("InputSim", "Unknown key name: " + keyName);
        }
    }

    private int getKeyCode(String keyName) {
        switch (keyName.toUpperCase()) {
            case "W":
                return KeyEvent.VK_W;
            case "A":
                return KeyEvent.VK_A;
            case "S":
                return KeyEvent.VK_S;
            case "D":
                return KeyEvent.VK_D;
            case "UP":
                return KeyEvent.VK_UP;
            case "DOWN":
                return KeyEvent.VK_DOWN;
            case "LEFT":
                return KeyEvent.VK_LEFT;
            case "RIGHT":
                return KeyEvent.VK_RIGHT;
            case "SPACE":
                return KeyEvent.VK_SPACE;
            case "ENTER":
                return KeyEvent.VK_ENTER;
            case "ESC":
                return KeyEvent.VK_ESCAPE;
            case "SHIFT":
                return KeyEvent.VK_SHIFT;
            case "CTRL":
                return KeyEvent.VK_CONTROL;
            case "ALT":
                return KeyEvent.VK_ALT;
            // Add more as needed or use reflection for full KeyEvent coverage
            default:
                if (keyName.length() == 1) {
                    char c = keyName.charAt(0);
                    return java.awt.event.KeyEvent.getExtendedKeyCodeForChar(c);
                }
                return -1;
        }
    }

    private void tapKey(int keyCode) {
        robot.keyPress(keyCode);
        robot.delay(30); // Slight delay for registration
        robot.keyRelease(keyCode);
    }

    public void pressKey(int keyCode) {
        if (robot != null)
            robot.keyPress(keyCode);
    }

    public void releaseKey(int keyCode) {
        if (robot != null)
            robot.keyRelease(keyCode);
    }
}
