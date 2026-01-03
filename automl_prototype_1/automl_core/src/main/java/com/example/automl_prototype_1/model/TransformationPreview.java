package com.example.automl_prototype_1.model;

import java.util.List;
import java.util.Map;

public class TransformationPreview {
    private List<Map<String, String>> previewRows;
    private String explanation;

    public TransformationPreview(List<Map<String, String>> previewRows, String explanation) {
        this.previewRows = previewRows;
        this.explanation = explanation;
    }

    public List<Map<String, String>> getPreviewRows() {
        return previewRows;
    }

    public String getExplanation() {
        return explanation;
    }
}
