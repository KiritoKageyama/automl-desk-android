package com.example.automl_prototype_1.rl.extensions;

import com.example.automl_prototype_1.rl.IEnvironment;

/**
 * A simple 5x5 Grid World environment for RL testing.
 * The agent starts at (0,0) and tries to reach (4,4).
 * Actions: 0=Up, 1=Down, 2=Left, 3=Right.
 * Reward: -0.1 per step, +10 at goal.
 */
public class GridWorldEnv implements IEnvironment {

    private int agentX;
    private int agentY;
    private final int gridSize = 5;
    private final int goalX = 4;
    private final int goalY = 4;

    public GridWorldEnv() {
        reset();
    }

    @Override
    public double[] reset() {
        agentX = 0;
        agentY = 0;
        return getState();
    }

    @Override
    public IEnvironment.StepResult step(int action) {
        // 0=Up, 1=Down, 2=Left, 3=Right
        int dx = 0, dy = 0;
        switch (action) {
            case 0:
                dy = -1;
                break;
            case 1:
                dy = 1;
                break;
            case 2:
                dx = -1;
                break;
            case 3:
                dx = 1;
                break;
        }

        // Move
        int newX = Math.max(0, Math.min(gridSize - 1, agentX + dx));
        int newY = Math.max(0, Math.min(gridSize - 1, agentY + dy));

        agentX = newX;
        agentY = newY;

        // Check Goal
        boolean done = (agentX == goalX && agentY == goalY);
        double reward = done ? 10.0 : -0.1;

        return new IEnvironment.StepResult(getState(), reward, done, "Pos: " + agentX + "," + agentY);
    }

    public double[] getState() {
        // Flattened state or coords
        return new double[] { (double) agentX, (double) agentY };
    }

    @Override
    public String render() {
        StringBuilder sb = new StringBuilder();
        for (int y = 0; y < gridSize; y++) {
            for (int x = 0; x < gridSize; x++) {
                if (x == agentX && y == agentY)
                    sb.append("A ");
                else if (x == goalX && y == goalY)
                    sb.append("G ");
                else
                    sb.append(". ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    @Override
    public void draw(Object graphicsContext) {
        if (!(graphicsContext instanceof javafx.scene.canvas.GraphicsContext)) {
            return;
        }

        javafx.scene.canvas.GraphicsContext gc = (javafx.scene.canvas.GraphicsContext) graphicsContext;
        double width = gc.getCanvas().getWidth();
        double height = gc.getCanvas().getHeight();

        double cellSize = Math.min(width, height) / gridSize;

        // Clear
        gc.clearRect(0, 0, width, height);

        // Draw Grid
        gc.setStroke(javafx.scene.paint.Color.LIGHTGRAY);
        for (int i = 0; i <= gridSize; i++) {
            gc.strokeLine(i * cellSize, 0, i * cellSize, gridSize * cellSize);
            gc.strokeLine(0, i * cellSize, gridSize * cellSize, i * cellSize);
        }

        // Draw Goal (Green)
        gc.setFill(javafx.scene.paint.Color.LIGHTGREEN);
        gc.fillRect(goalX * cellSize + 2, goalY * cellSize + 2, cellSize - 4, cellSize - 4);

        // Draw Agent (Blue)
        gc.setFill(javafx.scene.paint.Color.DODGERBLUE);
        gc.fillOval(agentX * cellSize + 5, agentY * cellSize + 5, cellSize - 10, cellSize - 10);
    }

    @Override
    public int getObservationSpaceSize() {
        return 2;
    }

    @Override
    public int getActionSpaceSize() {
        return 4;
    }

    @Override
    public String getName() {
        return "GridWorld-5x5";
    }
}
