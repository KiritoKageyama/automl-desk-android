package com.example.automl_prototype_1.dataprovider;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.LogService;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Loads image datasets from a directory structure.
 * Structure expected:
 * RootDir/
 * ClassA/
 * img1.jpg
 * img2.png
 * ClassB/
 * img3.jpg
 */
public class ImageDataProvider {

    public Dataset loadFromDirectory(File rootDir) throws IOException {
        if (!rootDir.exists() || !rootDir.isDirectory()) {
            throw new IllegalArgumentException("Invalid root directory: " + rootDir.getAbsolutePath());
        }

        List<String> filePaths = new ArrayList<>();
        List<Integer> labelsList = new ArrayList<>();
        Map<String, Integer> labelMap = new HashMap<>();
        int labelCounter = 0;

        File[] subDirs = rootDir.listFiles(File::isDirectory);
        if (subDirs == null) {
            throw new IOException("Failed to list directories in " + rootDir.getAbsolutePath());
        }

        for (File classDir : subDirs) {
            String className = classDir.getName();
            if (!labelMap.containsKey(className)) {
                labelMap.put(className, labelCounter++);
            }
            int labelId = labelMap.get(className);

            try (Stream<Path> paths = Files.walk(classDir.toPath())) {
                paths.filter(Files::isRegularFile)
                        .filter(p -> isImageFile(p.toString()))
                        .forEach(p -> {
                            filePaths.add(p.toAbsolutePath().toString());
                            labelsList.add(labelId);
                        });
            }
        }

        int[] labels = labelsList.stream().mapToInt(i -> i).toArray();
        LogService.info("ImageDataProvider", "Loaded " + filePaths.size() + " images from " + rootDir.getName()
                + ". Found " + labelMap.size() + " classes.");

        // Initialize placeholder features to match row count, otherwise validation
        // fails
        double[][] placeholders = new double[labels.length][0];

        // Initialize placeholder features to match row count

        return new Dataset(rootDir, placeholders, labels, filePaths, Dataset.DataType.IMAGE);
    }

    // We need a specific constructor in Dataset that accepts features AND
    // filePaths?
    // The previous one was Dataset(File, List<String>, int[], DataType). It set
    // features to [0][0].
    // I should create/use a constructor that allows setting features.

    private boolean isImageFile(String path) {
        String lower = path.toLowerCase();
        return lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png") || lower.endsWith(".bmp")
                || lower.endsWith(".gif");
    }
}
