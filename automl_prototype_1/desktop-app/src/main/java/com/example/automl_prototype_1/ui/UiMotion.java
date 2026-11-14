package com.example.automl_prototype_1.ui;

import javafx.animation.FadeTransition;
import javafx.animation.Interpolator;
import javafx.animation.ParallelTransition;
import javafx.animation.PauseTransition;
import javafx.animation.ScaleTransition;
import javafx.animation.SequentialTransition;
import javafx.animation.Timeline;
import javafx.animation.TranslateTransition;
import javafx.animation.KeyFrame;
import javafx.animation.KeyValue;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.layout.Pane;
import javafx.util.Duration;

/**
 * Lightweight motion helpers for website-like page and element transitions.
 */
public final class UiMotion {

    private UiMotion() {
    }

    public static void fadeSlideIn(Node node) {
        if (node == null) {
            return;
        }
        node.setOpacity(0);
        node.setTranslateY(18);
        FadeTransition fade = new FadeTransition(Duration.millis(320), node);
        fade.setFromValue(0);
        fade.setToValue(1);
        TranslateTransition slide = new TranslateTransition(Duration.millis(320), node);
        slide.setFromY(18);
        slide.setToY(0);
        slide.setInterpolator(Interpolator.EASE_OUT);
        new ParallelTransition(fade, slide).play();
    }

    public static void swapContent(Pane container, Parent next) {
        if (container == null || next == null) {
            return;
        }
        if (container.getChildren().isEmpty()) {
            container.getChildren().setAll(next);
            fadeSlideIn(next);
            return;
        }
        Node previous = container.getChildren().get(0);
        FadeTransition out = new FadeTransition(Duration.millis(140), previous);
        out.setFromValue(1);
        out.setToValue(0);
        out.setOnFinished(e -> {
            container.getChildren().setAll(next);
            fadeSlideIn(next);
        });
        out.play();
    }

    public static void staggerIn(Iterable<? extends Node> nodes, long baseDelayMs, long stepMs) {
        if (nodes == null) {
            return;
        }
        long delay = Math.max(0, baseDelayMs);
        for (Node node : nodes) {
            if (node == null) {
                continue;
            }
            node.setOpacity(0);
            node.setTranslateY(22);
            PauseTransition pause = new PauseTransition(Duration.millis(delay));
            pause.setOnFinished(e -> fadeSlideIn(node));
            pause.play();
            delay += Math.max(20, stepMs);
        }
    }

    public static void pulseOpacity(Node node, double min, double max, long millis) {
        if (node == null) {
            return;
        }
        Timeline timeline = new Timeline(
                new KeyFrame(Duration.ZERO, new KeyValue(node.opacityProperty(), min)),
                new KeyFrame(Duration.millis(millis), new KeyValue(node.opacityProperty(), max)));
        timeline.setAutoReverse(true);
        timeline.setCycleCount(Timeline.INDEFINITE);
        timeline.play();
    }

    public static void drift(Node node, double dx, double dy, long millis) {
        if (node == null) {
            return;
        }
        TranslateTransition t = new TranslateTransition(Duration.millis(millis), node);
        t.setByX(dx);
        t.setByY(dy);
        t.setAutoReverse(true);
        t.setCycleCount(TranslateTransition.INDEFINITE);
        t.setInterpolator(Interpolator.EASE_BOTH);
        t.play();
    }

    public static void pressScale(Node node) {
        if (node == null) {
            return;
        }
        ScaleTransition down = new ScaleTransition(Duration.millis(80), node);
        down.setToX(0.97);
        down.setToY(0.97);
        ScaleTransition up = new ScaleTransition(Duration.millis(120), node);
        up.setToX(1.0);
        up.setToY(1.0);
        new SequentialTransition(down, up).play();
    }
}
