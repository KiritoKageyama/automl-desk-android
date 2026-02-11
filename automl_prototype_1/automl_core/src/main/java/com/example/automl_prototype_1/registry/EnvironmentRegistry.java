package com.example.automl_prototype_1.registry;

import com.example.automl_prototype_1.rl.IEnvironment;
import com.example.automl_prototype_1.rl.extensions.GridWorldEnv;
import com.example.automl_prototype_1.service.LogService;
import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

/**
 * Registry to discover and provide available RL Environments.
 * Scans the 'com.example.automl_prototype_1.rl.extensions' package for custom
 * environments.
 */
public class EnvironmentRegistry {

    private static EnvironmentRegistry instance;
    private final List<Class<? extends IEnvironment>> availableEnvironments;

    private EnvironmentRegistry() {
        availableEnvironments = new ArrayList<>();
        // Always add default
        availableEnvironments.add(GridWorldEnv.class);
        refresh();
    }

    public static synchronized EnvironmentRegistry getInstance() {
        if (instance == null) {
            instance = new EnvironmentRegistry();
        }
        return instance;
    }

    /**
     * Rescans the extensions package for new environments.
     */
    public void refresh() {
        List<Class<? extends IEnvironment>> discovered = scanForEnvironments(
                "com.example.automl_prototype_1.rl.extensions");
        for (Class<? extends IEnvironment> envClass : discovered) {
            if (!availableEnvironments.contains(envClass)) {
                availableEnvironments.add(envClass);
                LogService.info("EnvRegistry", "Discovered new environment: " + envClass.getSimpleName());
            }
        }
    }

    public List<String> getEnvironmentNames() {
        List<String> names = new ArrayList<>();
        for (Class<? extends IEnvironment> cls : availableEnvironments) {
            names.add(cls.getSimpleName());
        }
        return names;
    }

    public IEnvironment createEnvironment(String name) {
        for (Class<? extends IEnvironment> cls : availableEnvironments) {
            if (cls.getSimpleName().equals(name)) {
                try {
                    return cls.getDeclaredConstructor().newInstance();
                } catch (Exception e) {
                    LogService.error("EnvRegistry",
                            "Failed to instantiate environment " + name + ": " + e.getMessage());
                }
            }
        }
        return null;
    }

    private List<Class<? extends IEnvironment>> scanForEnvironments(String packageName) {
        List<Class<? extends IEnvironment>> results = new ArrayList<>();
        try {
            String path = packageName.replace('.', '/');
            Enumeration<URL> resources = Thread.currentThread().getContextClassLoader().getResources(path);
            while (resources.hasMoreElements()) {
                URL resource = resources.nextElement();
                File directory = new File(resource.getFile());
                if (directory.exists() && directory.isDirectory()) {
                    File[] files = directory.listFiles();
                    if (files != null) {
                        for (File file : files) {
                            if (file.getName().endsWith(".class")) {
                                String className = packageName + "."
                                        + file.getName().substring(0, file.getName().length() - 6);
                                try {
                                    Class<?> cls = Class.forName(className);
                                    if (IEnvironment.class.isAssignableFrom(cls) && !cls.isInterface()
                                            && !java.lang.reflect.Modifier.isAbstract(cls.getModifiers())) {
                                        // Avoid duplicate checking here, handled in refresh
                                        // But GridWorldEnv is in this package too, so it might be found again.
                                        @SuppressWarnings("unchecked")
                                        Class<? extends IEnvironment> envCls = (Class<? extends IEnvironment>) cls;
                                        results.add(envCls);
                                    }
                                } catch (ClassNotFoundException e) {
                                    // Ignore
                                }
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            LogService.error("EnvRegistry", "Scan failed: " + e.getMessage());
        }
        return results;
    }
}
