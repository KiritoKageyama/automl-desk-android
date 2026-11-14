package com.example.automl_prototype_1;

import javafx.application.Application;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;
import java.net.URL;

public class DashboardApplication extends Application {

    // --- THEME MANAGEMENT ---
    private static final StringProperty themeProperty = new SimpleStringProperty("atelier");

    public static StringProperty themeProperty() {
        return themeProperty;
    }

    public static void setTheme(String theme) {
        if ("atelier".equals(theme) || "trading".equals(theme) || "light".equals(theme) || "dark".equals(theme)) {
            themeProperty.set(theme);
            System.out.println("Theme changed to: " + theme);
        }
    }

    @Override
    public void start(Stage stage) throws Exception {
        com.example.automl_prototype_1.service.GlobalExceptionHandlers.install();
        com.example.automl_prototype_1.gpu.GpuServerAuth.getInstance().ensureResolved();

        // --- Phase 6, Step 2: Background Resource Pre-Warming ---
        java.util.concurrent.CompletableFuture.runAsync(() -> {
            System.out.println("[PRE-WARM] Background Resource Initialization Started...");
            try {
                long start = System.currentTimeMillis();
                com.example.automl_prototype_1.gpu.HardwareAcceleratorService.isGpuAvailable();
                com.example.automl_prototype_1.gpu.GpuBatchService.getInstance();
                // Pre-load SelfCorrectionLoopService + ConstitutionalService
                com.example.automl_prototype_1.service.agentic.SelfCorrectionLoopService.getInstance();
                // Pre-seed Knowledge Graph
                com.example.automl_prototype_1.service.agentic.KnowledgeBaseService.getInstance()
                        .buildFromCurrentState();
                System.out.println("[PRE-WARM] Resources ready in " + (System.currentTimeMillis() - start) + "ms");
            } catch (Exception e) {
                System.err.println("[PRE-WARM] Non-critical failure: " + e.getMessage());
                com.example.automl_prototype_1.service.CrashReportService.getInstance()
                        .report("prewarm", e);
            }
        });

        // --- Screen-watcher: detect stuck training (Phase 9.2) ---
        // Starts in background; triggers SelfCorrectionLoopService.triggerCorrection()
        // on timeout
        com.example.automl_prototype_1.service.agentic.TrainingWatchdogService.getInstance().start();

        // --- Android remote monitor (LAN status API on :8088) ---
        try {
            com.example.automl_prototype_1.service.RemoteStatusServer.getInstance().start();
        } catch (Throwable t) {
            System.err.println("[RemoteStatus] Failed to start (app continues): " + t.getMessage());
            com.example.automl_prototype_1.service.CrashReportService.getInstance()
                    .report("RemoteStatus.start", t);
        }

        // --- Instantiate AppState ---
        com.example.automl_prototype_1.service.AppStateService appStateService = com.example.automl_prototype_1.service.AppStateService
                .getInstance();

        FXMLLoader fxmlLoader = new FXMLLoader(DashboardApplication.class.getResource("dashboard.fxml"));

        fxmlLoader.setControllerFactory(param -> {
            if (param == com.example.automl_prototype_1.controller.DashboardController.class) {
                try {
                    return new com.example.automl_prototype_1.controller.DashboardController(appStateService);
                } catch (Throwable t) {
                    System.err.println("CRITICAL FAILURE IN CONTROLLER FACTORY:");
                    t.printStackTrace();
                    throw new RuntimeException("Controller Creation Failed", t);
                }
            } else {
                try {
                    return param.getConstructor().newInstance();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        });

        Scene scene = new Scene(fxmlLoader.load());
        com.example.automl_prototype_1.controller.DashboardController dashboardController =
                fxmlLoader.getController();

        URL darkBaseUrl = getClass().getResource("/styles/dark_theme.css");
        URL atelierUrl = getClass().getResource("/styles/atelier_theme.css");
        URL tradingThemeUrl = getClass().getResource("/com/example/automl_prototype_1/css/trading-theme.css");
        URL lightThemeUrl = getClass().getResource("/com/example/automl_prototype_1/css/light-theme.css");

        if (darkBaseUrl != null) {
            scene.getStylesheets().add(darkBaseUrl.toExternalForm());
            System.out.println("[Theme] dark_theme.css applied as base.");
        } else {
            System.err.println("[Theme] WARNING: dark_theme.css not found");
        }

        final String atelierCss = atelierUrl != null ? atelierUrl.toExternalForm() : null;
        final String tradingThemeCss = tradingThemeUrl != null ? tradingThemeUrl.toExternalForm() : null;
        final String lightThemeCss = lightThemeUrl != null ? lightThemeUrl.toExternalForm() : null;

        Runnable applyTheme = () -> {
            scene.getStylesheets().removeIf(s ->
                    s.contains("atelier_theme") || s.contains("trading-theme") || s.contains("light-theme"));
            String current = themeProperty.get();
            if ("light".equals(current) && lightThemeCss != null) {
                scene.getStylesheets().add(lightThemeCss);
            } else if ("trading".equals(current) && tradingThemeCss != null) {
                scene.getStylesheets().add(tradingThemeCss);
            } else if (("atelier".equals(current) || "dark".equals(current)) && atelierCss != null) {
                scene.getStylesheets().add(atelierCss);
            } else if (atelierCss != null) {
                scene.getStylesheets().add(atelierCss);
            }
        };

        themeProperty().addListener((obs, oldTheme, newTheme) -> applyTheme.run());
        applyTheme.run();
        System.out.println("[Theme] Default atelier theme applied.");

        stage.setTitle("AutoML Studio — AI Atelier");
        stage.setMinWidth(1100);
        stage.setMinHeight(750);
        stage.setScene(scene);
        stage.show();

        javafx.application.Platform.runLater(
                com.example.automl_prototype_1.service.GlobalExceptionHandlers::installJavaFxThreadHandler);

        if (Boolean.getBoolean("automl.ui.smoke")
                || "true".equalsIgnoreCase(System.getenv("AUTOML_UI_SMOKE"))) {
            javafx.application.Platform.runLater(dashboardController::runViewSmokeTest);
        }
    }

    @Override
    public void stop() {
        try {
            com.example.automl_prototype_1.service.RemoteStatusServer.getInstance().stop();
        } catch (Exception ignored) {
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
