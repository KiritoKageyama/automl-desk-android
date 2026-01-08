@SuppressWarnings("module") module com.shadowtengu.automl_core {
    requires org.apache.commons.csv;
    requires smile.core;
    requires smile.data;
    requires smile.io;
    requires smile.math;
    requires org.slf4j;
    requires com.fasterxml.jackson.databind;
    requires java.net.http;
    requires transitive aparapi;

    exports com.example.automl_prototype_1.model;
    exports com.example.automl_prototype_1.service;
    exports com.example.automl_prototype_1.dataprovider;
    exports com.example.automl_prototype_1.algorithm;
    exports com.example.automl_prototype_1.classifier;
    exports com.example.automl_prototype_1.model.progress;
    exports com.example.automl_prototype_1.model.hyperparameter;
    exports com.example.automl_prototype_1.registry;
    exports com.example.automl_prototype_1.service.agentic;

    opens com.example.automl_prototype_1.model to com.fasterxml.jackson.databind;
    opens com.example.automl_prototype_1.service.agentic to com.fasterxml.jackson.databind;
    opens com.example.automl_prototype_1.classifier to aparapi;
}