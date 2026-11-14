package com.example.automl_prototype_1.registry;

import com.example.automl_prototype_1.classifier.genai.CustomScriptClassifier;
import com.example.automl_prototype_1.classifier.llm.GenericHuggingFaceClassifier;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.IClassifier;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ClassifierRegistryDynamicTest {

    @BeforeEach
    void resetDynamicModels() {
        AppStateService.getInstance().getDynamicModels().clear();
    }

    @Test
    void dynamicModelsAppearInClassifierNamesAndResolve() {
        AppStateService.getInstance().registerDynamicModel("MyBART", "facebook/bart-large");
        AppStateService.getInstance().registerDynamicModel("DynCls", "CLASS:FakeEnv");

        var names = ClassifierRegistry.getInstance().getClassifierNames();
        assertTrue(names.contains("MyBART"));
        assertTrue(names.contains("DynCls"));

        IClassifier hf = ClassifierRegistry.getInstance().getClassifier("MyBART");
        IClassifier cls = ClassifierRegistry.getInstance().getClassifier("DynCls");

        assertInstanceOf(GenericHuggingFaceClassifier.class, hf);
        assertEquals("MyBART", hf.getName());
        assertInstanceOf(CustomScriptClassifier.class, cls);
        assertEquals("DynCls", cls.getName());
    }
}
