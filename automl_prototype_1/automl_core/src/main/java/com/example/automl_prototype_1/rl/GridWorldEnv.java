package com.example.automl_prototype_1.rl;

import java.util.Random;

/**
 * GridWorldEnv — Phase 3 RL Lab enhancement.
 *
 * A configurable N×N grid world where an agent navigates from a start
 * cell to a goal cell, avoiding walls and collecting rewards.
 *
 * State : flat index row*N + col (one-hot encoded in stateVector)
 * Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
 * Rewards:
 * +10.0 on reaching the goal
 * -1.0 on hitting a wall or boundary
 * -0.1 per step (time penalty)
 */
public class GridWorldEnv implements IEnvironment {

    private static final int DEFAULT_SIZE = 8;
    private static final int DEFAULT_WALLS = 10;
    private static final int DEFAULT_MAX_STEPS = 200;

    private final int size;
    private final int maxSteps;
    private final boolean[][] walls;
    private final int goalRow;
    private final int goalCol;

    private int agentRow;
    private int agentCol;
    private int stepCount;
    private boolean done;
    private double lastReward;

    private static final int[][] DELTAS = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
    private static final String[] ACTION_NAMES = { "↑ UP", "↓ DOWN", "← LEFT", "→ RIGHT" };

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    public GridWorldEnv() {
        this(DEFAULT_SIZE, DEFAULT_WALLS, DEFAULT_MAX_STEPS);
    }

    public GridWorldEnv(int size, int numWalls, int maxSteps) {
        this.size = Math.max(4, size);
        this.maxSteps = maxSteps;
        this.goalRow = this.size - 1;
        this.goalCol = this.size - 1;
        this.walls = generateWalls(numWalls);
        reset();
    }

    // -------------------------------------------------------------------------
    // IEnvironment implementation
    // -------------------------------------------------------------------------

    @Override
    public double[] reset() {
        agentRow = 0;
        agentCol = 0;
        stepCount = 0;
        done = false;
        lastReward = 0;
        return stateVector();
    }

    @Override
    public IEnvironment.StepResult step(int action) {
        if (done) {
            return new IEnvironment.StepResult(stateVector(), 0, true, "Episode already done");
        }

        stepCount++;
        int newRow = agentRow + DELTAS[action][0];
        int newCol = agentCol + DELTAS[action][1];
        double reward;

        if (!inBounds(newRow, newCol) || walls[newRow][newCol]) {
            reward = -1.0; // Hit wall or boundary
        } else {
            agentRow = newRow;
            agentCol = newCol;
            if (agentRow == goalRow && agentCol == goalCol) {
                reward = +10.0;
                done = true;
            } else {
                reward = -0.1; // Time penalty
            }
        }

        if (stepCount >= maxSteps)
            done = true;
        lastReward = reward;
        return new IEnvironment.StepResult(stateVector(), reward, done, ACTION_NAMES[action]);
    }

    @Override
    public String render() {
        StringBuilder sb = new StringBuilder();
        sb.append("GridWorld ").append(size).append("×").append(size)
                .append("  Steps: ").append(stepCount).append("/").append(maxSteps)
                .append("  LastR: ").append(String.format("%.1f", lastReward)).append("\n");
        sb.append("┌").append("──".repeat(size)).append("┐\n");
        for (int r = 0; r < size; r++) {
            sb.append("│");
            for (int c = 0; c < size; c++) {
                if (r == agentRow && c == agentCol)
                    sb.append("🤖");
                else if (r == goalRow && c == goalCol)
                    sb.append("🎯");
                else if (walls[r][c])
                    sb.append("██");
                else
                    sb.append("· ");
            }
            sb.append("│\n");
        }
        sb.append("└").append("──".repeat(size)).append("┘\n");
        if (done)
            sb.append(agentRow == goalRow && agentCol == goalCol ? "✅ GOAL!" : "⏱ TIMEOUT");
        return sb.toString();
    }

    @Override
    public int getObservationSpaceSize() {
        return size * size;
    }

    @Override
    public int getActionSpaceSize() {
        return 4;
    }

    @Override
    public String getName() {
        return "GridWorld " + size + "×" + size;
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private double[] stateVector() {
        double[] s = new double[size * size];
        s[agentRow * size + agentCol] = 1.0;
        return s;
    }

    private boolean inBounds(int r, int c) {
        return r >= 0 && r < size && c >= 0 && c < size;
    }

    private boolean[][] generateWalls(int numWalls) {
        boolean[][] w = new boolean[size][size];
        Random rng = new Random(42);
        int placed = 0;
        while (placed < numWalls) {
            int r = rng.nextInt(size);
            int c = rng.nextInt(size);
            if ((r == 0 && c == 0) || (r == goalRow && c == goalCol))
                continue;
            if (!w[r][c]) {
                w[r][c] = true;
                placed++;
            }
        }
        return w;
    }
}
