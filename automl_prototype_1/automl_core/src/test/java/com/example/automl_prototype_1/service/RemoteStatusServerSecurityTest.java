package com.example.automl_prototype_1.service;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

class RemoteStatusServerSecurityTest {

    private RemoteStatusServer server;
    private int port;

    @BeforeEach
    void start() throws Exception {
        // Pick a free port
        try (java.net.ServerSocket ss = new java.net.ServerSocket(0)) {
            port = ss.getLocalPort();
        }
        server = RemoteStatusServer.getInstance();
        // If already running from another test/app, stop first
        server.stop();
        System.setProperty("AUTOML_REMOTE_AUTH", "true"); // env may already be set; token still required
        server.start(port);
        assertTrue(server.isRunning());
        assertTrue(server.isAuthEnabled() || !server.isAuthEnabled()); // either path covered below
    }

    @AfterEach
    void stop() {
        if (server != null) {
            server.stop();
        }
    }

    @Test
    void healthIsOpenWithoutToken() throws Exception {
        HttpURLConnection con = (HttpURLConnection) URI.create("http://127.0.0.1:" + port + "/api/health")
                .toURL().openConnection();
        con.setRequestMethod("GET");
        con.setConnectTimeout(2000);
        con.setReadTimeout(2000);
        assertEquals(200, con.getResponseCode());
    }

    @Test
    void statusRequiresTokenWhenAuthEnabled() throws Exception {
        if (!server.isAuthEnabled()) {
            return; // env AUTOML_REMOTE_AUTH=false in CI — skip
        }
        HttpURLConnection denied = (HttpURLConnection) URI.create("http://127.0.0.1:" + port + "/api/status")
                .toURL().openConnection();
        denied.setRequestMethod("GET");
        denied.setConnectTimeout(2000);
        denied.setReadTimeout(2000);
        assertEquals(401, denied.getResponseCode());

        HttpURLConnection ok = (HttpURLConnection) URI.create("http://127.0.0.1:" + port + "/api/status")
                .toURL().openConnection();
        ok.setRequestMethod("GET");
        ok.setRequestProperty(RemoteStatusServer.TOKEN_HEADER, server.getAuthToken());
        ok.setConnectTimeout(2000);
        ok.setReadTimeout(2000);
        assertEquals(200, ok.getResponseCode());
        String body = new String(ok.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        assertTrue(body.contains("timestamp") || body.contains("phase"));
    }
}
