package com.example.automl_prototype_1.gpu;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GpuServerAuthTest {

    @Test
    void resolveProvidesNonBlankTokenWhenAuthEnabled() {
        GpuServerAuth auth = GpuServerAuth.getInstance();
        auth.resetForTests();
        auth.ensureResolved();
        if (auth.isEnabled()) {
            assertNotNull(auth.getToken());
            assertFalse(auth.getToken().isBlank());
            assertEquals(GpuServerAuth.HEADER, "X-AutoML-Gpu-Token");
        }
    }
}
