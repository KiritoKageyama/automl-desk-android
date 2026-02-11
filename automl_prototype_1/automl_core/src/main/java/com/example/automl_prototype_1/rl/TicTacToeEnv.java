package com.example.automl_prototype_1.rl;

import java.util.Arrays;

public class TicTacToeEnv implements IEnvironment {

    private final double[] board; // 0=empty, 1=X (agent), -1=O (opponent)
    // private boolean agentTurn; // Always true in this simplified version
    private final int boardSize = 9;

    public TicTacToeEnv() {
        this.board = new double[boardSize];
        reset();
    }

    @Override
    public double[] reset() {
        Arrays.fill(board, 0);
        // agentTurn = true;
        return getObservation();
    }

    @Override
    public StepResult step(int action) {
        if (action < 0 || action >= boardSize) {
            return new StepResult(getObservation(), -10.0, true, "Invalid Action");
        }
        if (board[action] != 0) {
            return new StepResult(getObservation(), -10.0, true, "Illegal Move (Cell Occupied)");
        }

        // Agent Move
        board[action] = 1.0;
        if (checkWin(1.0)) {
            return new StepResult(getObservation(), 10.0, true, "Agent Won!");
        }
        if (isFull()) {
            return new StepResult(getObservation(), 0.0, true, "Draw");
        }

        // Opponent Move (Random for now)
        int opponentMove = makeRandomMove();
        if (opponentMove != -1) {
            board[opponentMove] = -1.0;
            if (checkWin(-1.0)) {
                return new StepResult(getObservation(), -10.0, true, "Agent Lost");
            }
        }

        if (isFull()) {
            return new StepResult(getObservation(), 0.0, true, "Draw");
        }

        return new StepResult(getObservation(), 0.0, false, "Continue");
    }

    private int makeRandomMove() {
        for (int i = 0; i < boardSize; i++) {
            if (board[i] == 0)
                return i; // Dumb opponent: picks first available
        }
        return -1;
    }

    private boolean checkWin(double player) {
        // Rows
        for (int i = 0; i < 3; i++) {
            if (board[i * 3] == player && board[i * 3 + 1] == player && board[i * 3 + 2] == player)
                return true;
        }
        // Cols
        for (int i = 0; i < 3; i++) {
            if (board[i] == player && board[i + 3] == player && board[i + 6] == player)
                return true;
        }
        // Diagonals
        if (board[0] == player && board[4] == player && board[8] == player)
            return true;
        if (board[2] == player && board[4] == player && board[6] == player)
            return true;

        return false;
    }

    private boolean isFull() {
        for (double v : board) {
            if (v == 0)
                return false;
        }
        return true;
    }

    private double[] getObservation() {
        return Arrays.copyOf(board, boardSize);
    }

    @Override
    public String render() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double v = board[i * 3 + j];
                char c = v == 1.0 ? 'X' : (v == -1.0 ? 'O' : '.');
                sb.append(c).append(" ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    @Override
    public int getObservationSpaceSize() {
        return boardSize;
    }

    @Override
    public int getActionSpaceSize() {
        return boardSize;
    }

    @Override
    public String getName() {
        return "Tic-Tac-Toe";
    }
}
