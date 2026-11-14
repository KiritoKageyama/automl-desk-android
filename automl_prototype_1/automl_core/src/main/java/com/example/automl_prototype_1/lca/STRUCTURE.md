# Package: LCA (Layered Cognitive Agent)

## Feature Context
The **Layered Cognitive Agent (LCA)** is the autonomous "Ghost in the Shell" of the AutoML system. It is designed to observe the user's screen, understand context, and execute actions to automate tasks or assist the user. It follows a 4-Layer Memory architecture (OS, App, Domain, Global) and uses a fast perception-action loop.

## Architecture Position
**Core / Agent Layer**
- **Upstream**: Receives screen data from the OS (via `PerceptionService`) and loop commands from the UI (`AgentDebugViewController`).
- **Downstream**: Sends high-level "See" snapshots to the Python "Brain" (`gpu_server.py`) and executes low-level "Act" commands via `ActionDispatcher`.

## File Map
- **`LcaService.java`**: The central orchestrator. Runs the 10Hz "Fast Loop", synchronizes perception and action, and communicates with the Python backend.
- **`perception/PerceptionService.java`**: The "Eyes". Captures screen screenshots and runs OCR (Tesseract) to extract text nodes.
- **`actor/ActionDispatcher.java`**: The "Hands". Receives action commands (Click, Type) from the Brain and executes them using Java AWT Robot.
