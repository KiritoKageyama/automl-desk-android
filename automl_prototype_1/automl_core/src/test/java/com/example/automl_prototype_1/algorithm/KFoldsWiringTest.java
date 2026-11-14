package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.service.AppStateService;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class KFoldsWiringTest {

    @Test
    void resolveKFoldsHonorsAppState() {
        AppStateService.getInstance().setKFolds(10);
        assertEquals(10, AbstractFeatureSelectionAlgorithm.resolveKFolds(5));

        AppStateService.getInstance().setKFolds(3);
        GeneticAlgorithm ga = new GeneticAlgorithm(10, 20);
        assertEquals(3, ga.getKFolds());
    }

    @Test
    void resolveKFoldsClamps() {
        AppStateService.getInstance().setKFolds(100);
        assertEquals(20, AbstractFeatureSelectionAlgorithm.resolveKFolds(5));
        AppStateService.getInstance().setKFolds(1);
        assertEquals(5, AbstractFeatureSelectionAlgorithm.resolveKFolds(5));
    }
}
