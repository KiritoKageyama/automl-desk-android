package com.example.automl_prototype_1.experiment;

import com.example.automl_prototype_1.model.Dataset;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class SyntheticDataGenerator {

    public static Dataset generate(int numInstances, int numFeatures, int numInformative, boolean saveToFile) {
        double[][] features = new double[numInstances][numFeatures];
        int[] labels = new int[numInstances];
        Random random = new Random(42);

        // Generate features
        for (int i = 0; i < numInstances; i++) {
            // Class label (0 or 1)
            labels[i] = random.nextBoolean() ? 1 : 0;

            for (int j = 0; j < numFeatures; j++) {
                if (j < numInformative) {
                    // Informative feature: Correlated with label + noise
                    double signal = (labels[i] == 1) ? 1.0 : -1.0;
                    features[i][j] = signal + random.nextGaussian() * 0.5;

                    // Discretize some informative features to be "low cardinality"
                    if (j % 2 == 0) {
                        features[i][j] = Math.round(features[i][j]);
                    }
                } else {
                    // Noise feature: Random Gaussian
                    features[i][j] = random.nextGaussian();
                }
            }
        }

        // Create CSV content
        StringBuilder sb = new StringBuilder();
        // Header
        for (int j = 0; j < numFeatures; j++) {
            sb.append("F").append(j).append(",");
        }
        sb.append("Class\n");

        // Data
        for (int i = 0; i < numInstances; i++) {
            for (int j = 0; j < numFeatures; j++) {
                sb.append(String.format("%.4f", features[i][j])).append(",");
            }
            sb.append(labels[i]).append("\n");
        }

        if (saveToFile) {
            try (FileWriter writer = new FileWriter("synthetic_ablation_data.csv")) {
                writer.write(sb.toString());
                System.out.println("Saved synthetic dataset to synthetic_ablation_data.csv");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        // Return Dataset object (Assuming constructor or loader exists, for now we load
        // from file in runner to be safe)
        return null;
    }
}
