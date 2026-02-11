package com.example.automl_prototype_1.gpu;

/**
 * Wrapper for serialized training data (JSON Text or Arrow Binary).
 */
public class SerializedData {
    private final byte[] body;
    private final String contentType;
    private final boolean isBinary;

    public SerializedData(byte[] body, String contentType, boolean isBinary) {
        this.body = body;
        this.contentType = contentType;
        this.isBinary = isBinary;
    }

    public static SerializedData fromJson(String json) {
        return new SerializedData(json.getBytes(java.nio.charset.StandardCharsets.UTF_8), "application/json", false);
    }

    public static SerializedData fromBinary(byte[] data, String contentType) {
        return new SerializedData(data, contentType, true);
    }

    public byte[] getBody() {
        return body;
    }

    public String getContentType() {
        return contentType;
    }

    public boolean isBinary() {
        return isBinary;
    }
}
