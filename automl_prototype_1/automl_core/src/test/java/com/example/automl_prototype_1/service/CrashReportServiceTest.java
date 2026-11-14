package com.example.automl_prototype_1.service;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class CrashReportServiceTest {

    @Test
    void reportPersistsFileAndIncrementsCount() throws Exception {
        CrashReportService svc = CrashReportService.getInstance();
        int before = svc.getReportCount();
        Path file = svc.report("unit-test", new IllegalStateException("boom-test"));
        assertNotNull(file);
        assertTrue(Files.isRegularFile(file));
        String body = Files.readString(file);
        assertTrue(body.contains("boom-test"));
        assertTrue(body.contains("unit-test"));
        assertEquals(before + 1, svc.getReportCount());
    }
}
