package com.example.automl_prototype_1;

import javafx.application.Application;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;
import java.net.URL;

public class DashboardApplication extends Application {

    // --- THEME MANAGEMENT IS NOW HERE ---
    private static final StringProperty themeProperty = new SimpleStringProperty("trading"); // Default to trading

    public static StringProperty themeProperty() {
        return themeProperty;
    }

    public static void setTheme(String theme) {
        if ("trading".equals(theme) || "light".equals(theme)) {
            themeProperty.set(theme);
            System.out.println("Theme changed to: " + theme);
        }
    }
    // ------------------------------------

    @Override
    public void start(Stage stage) throws Exception {
        // --- Clean Architecture: Instantiate Service ---
        com.example.automl_prototype_1.service.AppStateService appStateService = com.example.automl_prototype_1.service.AppStateService
                .getInstance();

        FXMLLoader fxmlLoader = new FXMLLoader(DashboardApplication.class.getResource("dashboard.fxml"));

        // --- Dependency Injection ---
        fxmlLoader.setControllerFactory(param -> {
            if (param == com.example.automl_prototype_1.controller.DashboardController.class) {
                return new com.example.automl_prototype_1.controller.DashboardController(appStateService);
            } else {
                try {
                    return param.getConstructor().newInstance();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        });

        Scene scene = new Scene(fxmlLoader.load());

        URL tradingThemeUrl = getClass().getResource("/com/example/automl_prototype_1/css/trading-theme.css");
        URL lightThemeUrl = getClass().getResource("/com/example/automl_prototype_1/css/light-theme.css");

        if (tradingThemeUrl == null || lightThemeUrl == null) {
            System.err.println("CRITICAL: Theme CSS files not found. Check paths. " + tradingThemeUrl);
        } else {
            final String tradingThemeCss = tradingThemeUrl.toExternalForm();
            final String lightThemeCss = lightThemeUrl.toExternalForm();

            // The listener now listens to the property within this class
            themeProperty().addListener((obs, oldTheme, newTheme) -> {
                scene.getStylesheets().clear();
                if ("light".equals(newTheme)) {
                    scene.getStylesheets().add(lightThemeCss);
                } else {
                    // Default to trading for any other value
                    scene.getStylesheets().add(tradingThemeCss);
                }
            });

            // Force initial application of the theme
            String currentTheme = themeProperty.get();
            if ("light".equals(currentTheme)) {
                scene.getStylesheets().add(lightThemeCss);
            } else {
                scene.getStylesheets().add(tradingThemeCss);
            }
        }

        stage.setTitle("AutoML Dashboard - Quantitative Trading Edition");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}