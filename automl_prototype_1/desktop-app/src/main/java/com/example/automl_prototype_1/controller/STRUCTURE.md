# Package: Controller

## Feature Context
This package contains the **View Controllers** for the JavaFX Desktop Application. It handles user interactions, manages the UI state, and updates the view based on data from the backend services.

## Architecture Position
**Presentation / UI Layer**
- **Upstream**: Driven by User Input (Clicks, Typing).
- **Downstream**: Calls into `service` layer to trigger actions; observes `LcaService` for updates.

## File Map
- **`DashboardController.java`**: The main shell of the application. Manages navigation between different tabs (Neural Forge, Creator Lab, Settings, LCA Agent).
- **`NeuralForgeViewController.java`**: Controls the "Neural Forge" view, where users build and visualize neural network architectures.
- **`AgentDebugViewController.java`**: Provides the debug interface for the Layered Cognitive Agent, showing the live stream of the "Mind's Eye" and logs of its thought process.
- **`DataVisualizationViewController.java`**: Handles plotting of charts and graphs to visualize dataset distributions and training results.
- **`CreatorLabViewController.java`**: The interface for experimenting with new algorithm combinations and custom pipelines.
