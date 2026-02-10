package com.example.automl_prototype_1.service.agentic;

import java.io.IOException;
import java.util.List;

/**
 * Service to aggregate project context (Project Tree, File Contents) for the
 * LLM.
 * Acts as the "Eyes" of the self-coding system.
 */
public class ProjectContextService {

    private final CodeModificationService codeModService;

    public ProjectContextService() {
        this.codeModService = new CodeModificationService();
    }

    /**
     * Generates a concise tree structure of the project.
     * Useful for the LLM to know where files are located.
     */
    public String getProjectStructure() {
        try {
            List<String> files = codeModService.listProjectFiles();
            StringBuilder sb = new StringBuilder("Project Structure:\n");
            // Simple list for now, could be tree-formatted later if needed
            for (String f : files) {
                sb.append("- ").append(f).append("\n");
            }
            return sb.toString();
        } catch (IOException e) {
            return "Error listing project files: " + e.getMessage();
        }
    }

    /**
     * Retrieves specific file contents for the LLM to read.
     */
    public String getFileContext(List<String> filePaths) {
        StringBuilder sb = new StringBuilder();
        for (String path : filePaths) {
            sb.append("\n--- File: ").append(path).append(" ---\n");
            try {
                String content = codeModService.readFile(path);
                // Limit content size if huge?
                if (content.length() > 50000) {
                    sb.append("(File truncated, >50k chars)\n");
                    sb.append(content.substring(0, 5000));
                } else {
                    sb.append(content);
                }
            } catch (Exception e) {
                sb.append("(Error reading file: ").append(e.getMessage()).append(")\n");
            }
            sb.append("\n--- End of ").append(path).append(" ---\n");
        }
        return sb.toString();
    }
}
