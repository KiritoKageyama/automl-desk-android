# Creator Lab (AI Architect) — User Guide

## What is Creator Lab?

Creator Lab lets you describe an RL environment in plain English, and the AI generates a complete Java implementation. You can then compile it directly from the UI and train an agent on it using the RL Lab or Visualizer.

## Prerequisites

- **Ollama** must be running locally (default: `http://localhost:11434`)
- A code-capable model must be loaded (e.g., `codellama`, `deepseek-coder`, `mistral`)
- Configure the Ollama URL in **Settings → AI Assistant Configuration**

## Step-by-Step Workflow

### 1. Open Creator Lab
Click **"Creator Lab (AI Architect)"** in the sidebar navigation.

### 2. Enter Your Prompt
Describe the environment you want to create. Be specific about:
- **Game rules** — what constitutes a win/loss
- **State representation** — what the agent observes
- **Actions** — what the agent can do
- **Rewards** — how the agent is scored

**Example prompts:**
- `"Create a Snake Game environment where the agent controls a snake on a 10x10 grid"`
- `"Create a simple maze where the agent must find the exit"`
- `"Create a stock trading environment with buy/sell/hold actions"`

### 3. Set the Class Name
Enter a name for your new environment class (e.g., `SnakeGameEnv`). If left blank, it defaults to `CustomEnv`.

### 4. Click "Generate"
The AI will generate a complete Java class that implements the `IEnvironment` interface. This includes:
- `reset()` — initializes the environment
- `step(int action)` — advances one timestep
- `getObservation()` — returns current state
- `isTerminated()` — checks if episode is done
- `draw(GraphicsContext gc)` — visual rendering

### 5. Review the Code
Always review the generated code before compiling. Check for:
- Correct import statements
- Proper interface implementation
- Sensible reward logic
- No dangerous operations (file I/O, network calls)

### 6. Click "Compile"
The dynamic compiler runs on the generated source file. If successful:
- The `.class` file is placed in `target/classes/`
- The new environment is registered for use

**Note:** A JVM restart may be required for the new class to appear in dropdowns due to ClassLoader constraints.

### 7. Train Your Agent
Go to **RL Lab** or **Visualizer** to select your new environment and train an RL agent on it.

## Architecture

```
[You] → Prompt → [AgenticArchitectService] → [Ollama LLM]
                                                   ↓
                                           Generated Java Code
                                                   ↓
                            [DynamicCompilerService] → Compiled .class
                                                   ↓
                            [RL Lab / Visualizer] → Train Agent
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No Java Compiler available" | Ensure you're running with a JDK (not JRE) |
| Compilation fails | Review error messages in the log area; common issues are missing imports or interface method mismatches |
| New class not in dropdown | Restart the application (JVM ClassLoader limitation) |
| AI generates bad code | Try a more specific prompt, or manually edit the code before compiling |

## Tips

- **Start simple.** Grid-based environments (like the built-in `GridWorldEnv`) work best.
- **Use the reference.** Look at `GridWorldEnv.java` in `rl/extensions/` for a working example.
- **Iterate.** Generate, review, edit manually, then compile.
