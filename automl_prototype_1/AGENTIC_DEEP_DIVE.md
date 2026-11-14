# Agentic AI & Advanced Features Deep Dive

**Date:** 2026-02-14
**Scope:** Agentic Architecture, LLM/SLM Integration, Reinforcement Learning, Visualization.

## 1. Executive Summary
The application features a sophisticated **Hybrid AI Architecture** that combines:
1.  **Layered Cognitive Agent (LCA)**: A "Brain Stem" loop for real-time perception and action.
2.  **Hierarchical LLM System**: Orchestrated usage of Large Language Models (Ollama: Llama3, Mistral) and Small Language Models (SLM: TinyLlama via Python) for different cognitive loads.
3.  **Reinforcement Learning (RL)**: Q-Learning agents capable of "Kickstarting" via Behavioral Cloning (human demonstration).
4.  **Constitutional Safety**: A rule-based guardrail system that audits both decisions and generated code.

---

## 2. Autonomous Agent (LCA) Architecture

The **LCA Service** (`com.example.automl_prototype_1.lca`) functions as the central nervous system of the agent.

### Core Loop (The "Brain Stem")
The agent runs on a high-frequency loop (target 10Hz) to maintain real-time responsiveness:
1.  **Perception**: `PerceptionService` captures the environment state (e.g., screen snapshots, game state).
2.  **Cognition (The "Think" Step)**:
    -   The snapshot is serialized and sent to the **Python Agent Server** (`gpu_server.py` on port 5006).
    -   This allows heavy inference (Vision Models, LLMs) to run on GPU without blocking the Java UI thread.
3.  **Action**: `ActionDispatcher` executes the decision using `java.awt.Robot`.
    -   **Capabilities**: Mouse Movement, Clicking (via Bounding Box center calculation), Key Typing.

### Active Exploration
-   **Service**: `ActiveExplorationService`
-   **Function**: The agent navigates game menus or environments autonomously to build a state map, driven by curiosity or specific "Profiles" (e.g., "WASD", "TREX").

---

## 3. LLM & SLM Integration

The system employs a **Tiered Intelligence Strategy** to balance performance and capability.

### Tier 1: Local LLMs (Ollama)
-   **Service**: `OllamaService`
-   **Models**: Auto-detects and prioritizes: *Deepseek > Qwen > Llama3 > Mistral*.
-   **Lifecycle**: "Start-on-Demand, Stop-on-Idle" (5-minute timeout) to conserve user resources.
-   **RAG (Retrieval Augmented Generation)**: Injects `ProjectContextService` (project structure, file contents) into prompts to give the LLM "self-awareness" of the codebase.

### Tier 2: Embedded SLMs (Python Bridge)
-   **Service**: `PythonExecutionService` -> `gpu_server.py`
-   **Model**: Likely **TinyLlama** or similar quantized models running entirely local/embedded.
-   **Purpose**: Fast, low-latency tasks (e.g., "Recall Memories", simple chat).
-   **Memory System**: The Python layer supports querying a vector store (`queryMemory`) to retrieve relevant past contexts.

### Generative Coding & Safety
-   **Feature**: `generateClassifierCode(description)`
-   **Workflow**:
    1.  Injects `IClassifier` interface contract.
    2.  Injects `LogisticRegression.java` as a style example.
    3.  Generates code via LLM.
    4.  **Constitutional Audit**: Passes code through `ConstitutionalService` to check for banned patterns (e.g., `System.exit`, malicious imports) before accepting it.
    5.  **Self-Correction**: If the audit fails, the error is fed back to the LLM to regenerate a compliant version.

---

## 4. Reinforcement Learning (RL) Systems

The **RL Module** (`com.example.automl_prototype_1.rl`) provides adaptive learning capabilities.

### Algorithms
-   **Tabular Q-Learning**: Implemented in `TabularQLearningAgent`.
-   **Action Space**: Discrete (mapped to keys like WASD or Arrow keys).

### Behavioral Cloning ("Kickstart")
-   **Service**: `BehavioralCloningService`
-   **Innovation**: Solves the "Cold Start" problem in RL.
    1.  **Record**: User plays the game/environment manually.
    2.  **Learn**: System trains a supervised model (KNN/RF) on (State -> Action) pairs.
    3.  **Kickstart**: The RL agent pre-fills its Q-Table using this "Cloned Policy" before starting independent exploration.

### Introspection
-   **Service**: `IntrospectionService`
-   **Logic**: Monitors the "Reward Trend" over a sliding window (50 steps).
-   **Detection**: Identifies "Degrading Performance" or "Concept Drift" and triggers alerts/advice in the UI.

---

## 5. Visualization & User Experience

The **Desktop App** (`com.example.automl_prototype_1.desktop_app`) provides deep observability into the AI's mind.

### RL Visualizer (`RlVisualizerController`)
-   **Real-Time Canvas**: Renders the environment state (e.g., GridWorld).
-   **Insight Stream**: Displays the "Inner Monologue" or "Knowledge Base" rules the agent is learning (`KnowledgeBaseService`).
-   **Interactive Training**: Users can Start/Stop training, Record demonstrations, and Edit learned insights manually.

### "Mind's Eye"
-   **Feature**: The `LcaService` exposes an `AgentState` stream containing:
    -   The Screenshot (Vision)
    -   The Thought Process (JSON decision)
    -   Timestamp
-   This allows the user to see exactly *what* the agent saw and *why* it acted.

---

## 6. Formalization & Safety (Gen AI)

### Constitutional AI (`ConstitutionalService`)
-   **Concept**: A "Rule of Law" for the AI.
-   **Mechanism**: A JSON-based Constitution (`constitution.json`) defines strict rules.
    -   **VETO**: Blocks actions/algorithms (e.g., "Don't use KNN on >1M rows").
    -   **CODE_BAN**: Regex-based blocking of unsafe code patterns.
-   **Modes**: "Strict Mode" enforces resource limits (RAM/CPU) aggressively to prevent system freezes.

### Domain Guards (`DomainGuardService`)
-   **Function**: Validates LLM prompts *before* execution.
-   **Purpose**: Ensures the AI stays within the "Computer Science / AutoML" domain and rejects unrelated or unsafe queries.
