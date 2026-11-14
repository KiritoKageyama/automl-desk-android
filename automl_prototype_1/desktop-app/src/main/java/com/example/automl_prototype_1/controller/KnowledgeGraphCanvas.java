package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.model.KnowledgeEdge;
import com.example.automl_prototype_1.model.KnowledgeNode;
import com.example.automl_prototype_1.model.KnowledgeNode.NodeType;
import com.example.automl_prototype_1.model.KnowledgeEdge.Relationship;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.paint.Color;
import javafx.scene.paint.CycleMethod;
import javafx.scene.paint.LinearGradient;
import javafx.scene.paint.RadialGradient;
import javafx.scene.paint.Stop;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.TextAlignment;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;

/**
 * Physics-based interactive Knowledge Graph Canvas.
 *
 * Physics model — Verlet integration with:
 * - Coulomb repulsion between all node pairs (O(n²))
 * - Hooke spring attraction along edges
 * - Gravitational center pull (prevents unbounded drift)
 * - Velocity damping (simulates air resistance)
 * - Hard boundary walls with elastic rebound
 * - Node mass proportional to degree (hub nodes are heavier)
 * - Collision avoidance (nodes push away when overlapping)
 *
 * Visual features:
 * - 13 node types with unique colors and icons
 * - Directed arrows with bezier curves for self-loops
 * - Edge label shown on hover/select
 * - Radial glow animation on recently-active nodes
 * - Pulse ring animation on selected node
 * - HUD: node count, edge count, zoom%, FPS
 * - Right-click context menu (delete / pin / inspect)
 * - Double-click to pin/unpin a node
 */
public class KnowledgeGraphCanvas extends Canvas {

    // =========================================================================
    // Data
    // =========================================================================
    private List<KnowledgeNode> nodes = new ArrayList<>();
    private List<KnowledgeEdge> edges = new ArrayList<>();

    // =========================================================================
    // Camera
    // =========================================================================
    private double offsetX = 0, offsetY = 0;
    private double zoom = 1.0;
    private static final double MIN_ZOOM = 0.15;
    private static final double MAX_ZOOM = 8.0;

    // =========================================================================
    // Interaction state
    // =========================================================================
    private KnowledgeNode draggedNode = null;
    private KnowledgeNode selectedNode = null;
    private KnowledgeEdge selectedEdge = null;
    private KnowledgeNode hoveredNode = null;
    private boolean isDraggingCanvas = false;
    private double lastMouseX, lastMouseY;
    private long lastRenderNs = 0;
    private int fpsCounter = 0, fps = 0;
    private long fpsTimer = 0;

    // =========================================================================
    // Rendering constants
    // =========================================================================
    private static final double NODE_RADIUS = 24;
    private static final double GLOW_RADIUS = 46;
    private static final long GLOW_WINDOW_MS = 3500;
    private static final double PULSE_MAX_R = 40;

    // =========================================================================
    // Physics constants
    // =========================================================================
    private static final double REPULSION_K = 12000.0; // Coulomb constant
    private static final double SPRING_K = 0.003; // Hooke constant
    private static final double REST_LENGTH = 180.0; // Spring rest length (px)
    private static final double DAMPING = 0.82; // Velocity damping per step
    private static final double GRAVITY_K = 0.0004; // Center gravity
    private static final double COLLISION_R = NODE_RADIUS * 2.2;
    private static final double WALL_MARGIN = 60.0;
    private static final double WALL_BOUNCE = 0.35; // Elasticity on wall collision
    private static final double MASS_BASE = 1.0;
    private static final double MASS_PER_EDGE = 0.4;

    // =========================================================================
    // Callbacks
    // =========================================================================
    private BiConsumer<KnowledgeNode, List<KnowledgeEdge>> onNodeSelected;
    private BiConsumer<KnowledgeEdge, KnowledgeNode[]> onEdgeSelected;

    // =========================================================================
    // Constructor
    // =========================================================================
    public KnowledgeGraphCanvas() {
        super();
        setOnMousePressed(this::handleMousePressed);
        setOnMouseDragged(this::handleMouseDragged);
        setOnMouseReleased(this::handleMouseReleased);
        setOnMouseMoved(this::handleMouseMoved);
        setOnScroll(this::handleScroll);
    }

    // =========================================================================
    // Public API
    // =========================================================================
    public void setData(List<KnowledgeNode> nodes, List<KnowledgeEdge> edges) {
        this.nodes = nodes != null ? nodes : new ArrayList<>();
        this.edges = edges != null ? edges : new ArrayList<>();
        // Assign masses based on degree
        for (KnowledgeNode node : this.nodes) {
            long degree = this.edges.stream()
                    .filter(e -> e.getSourceId().equals(node.getId()) || e.getTargetId().equals(node.getId()))
                    .count();
            node.setMass(MASS_BASE + degree * MASS_PER_EDGE);
        }
        // Scatter unpositioned nodes around canvas center
        double cx = getWidth() > 0 ? getWidth() / 2 : 400;
        double cy = getHeight() > 0 ? getHeight() / 2 : 300;
        for (KnowledgeNode node : this.nodes) {
            if (node.getX() == 0 && node.getY() == 0) {
                double angle = Math.random() * 2 * Math.PI;
                double dist = 80 + Math.random() * 200;
                node.setX(cx + Math.cos(angle) * dist);
                node.setY(cy + Math.sin(angle) * dist);
            }
        }
    }

    public void setOnNodeSelected(BiConsumer<KnowledgeNode, List<KnowledgeEdge>> cb) {
        this.onNodeSelected = cb;
    }

    public void setOnEdgeSelected(BiConsumer<KnowledgeEdge, KnowledgeNode[]> cb) {
        this.onEdgeSelected = cb;
    }

    public KnowledgeNode getSelectedNode() {
        return selectedNode;
    }

    public KnowledgeEdge getSelectedEdge() {
        return selectedEdge;
    }

    public void panToNode(KnowledgeNode node) {
        offsetX = getWidth() / 2 - node.getX() * zoom;
        offsetY = getHeight() / 2 - node.getY() * zoom;
        render();
    }

    public void selectNode(KnowledgeNode node) {
        selectedNode = node;
        if (onNodeSelected != null) {
            List<KnowledgeEdge> connected = edges.stream()
                    .filter(e -> e.getSourceId().equals(node.getId()) || e.getTargetId().equals(node.getId()))
                    .toList();
            onNodeSelected.accept(node, connected);
        }
        render();
    }

    public void centerView() {
        if (nodes.isEmpty())
            return;
        double minX = nodes.stream().mapToDouble(KnowledgeNode::getX).min().orElse(0);
        double maxX = nodes.stream().mapToDouble(KnowledgeNode::getX).max().orElse(0);
        double minY = nodes.stream().mapToDouble(KnowledgeNode::getY).min().orElse(0);
        double maxY = nodes.stream().mapToDouble(KnowledgeNode::getY).max().orElse(0);
        double graphW = maxX - minX;
        double graphH = maxY - minY;
        zoom = Math.min(MAX_ZOOM, Math.min(
                (getWidth() - 120) / Math.max(graphW, 1),
                (getHeight() - 120) / Math.max(graphH, 1)));
        offsetX = getWidth() / 2 - (minX + graphW / 2) * zoom;
        offsetY = getHeight() / 2 - (minY + graphH / 2) * zoom;
        render();
    }

    // =========================================================================
    // Physics — Verlet integration
    // =========================================================================
    public void layoutStep() {
        if (nodes == null || nodes.size() < 2)
            return;
        double cx = getWidth() / 2 / zoom;
        double cy = getHeight() / 2 / zoom;

        for (int i = 0; i < nodes.size(); i++) {
            KnowledgeNode a = nodes.get(i);
            if (a.isPinned() || a == draggedNode)
                continue;

            double fx = 0, fy = 0;
            double mass = a.getMass();

            // 1. Coulomb repulsion — every other node repels
            for (int j = 0; j < nodes.size(); j++) {
                if (i == j)
                    continue;
                KnowledgeNode b = nodes.get(j);
                double dx = a.getX() - b.getX();
                double dy = a.getY() - b.getY();
                double dist2 = dx * dx + dy * dy;
                if (dist2 < 1e-4) {
                    fx += Math.random() * 2 - 1;
                    fy += Math.random() * 2 - 1;
                    continue;
                }
                double dist = Math.sqrt(dist2);
                double force = REPULSION_K / (dist2 * mass);
                fx += (dx / dist) * force;
                fy += (dy / dist) * force;

                // 2. Collision avoidance — extra push when overlapping
                if (dist < COLLISION_R) {
                    double overlap = COLLISION_R - dist;
                    fx += (dx / dist) * overlap * 1.5;
                    fy += (dy / dist) * overlap * 1.5;
                }
            }

            // 3. Hooke spring — along edges only
            for (KnowledgeEdge edge : edges) {
                KnowledgeNode other = null;
                if (edge.getSourceId().equals(a.getId()))
                    other = findNode(edge.getTargetId());
                else if (edge.getTargetId().equals(a.getId()))
                    other = findNode(edge.getSourceId());
                if (other == null)
                    continue;

                double dx = other.getX() - a.getX();
                double dy = other.getY() - a.getY();
                double dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1.0);
                double displacement = dist - REST_LENGTH;
                double force = SPRING_K * displacement / mass;
                fx += (dx / dist) * force;
                fy += (dy / dist) * force;
            }

            // 4. Gravity toward canvas center
            double viewCx = (getWidth() / 2 - offsetX) / zoom;
            double viewCy = (getHeight() / 2 - offsetY) / zoom;
            fx += (viewCx - a.getX()) * GRAVITY_K;
            fy += (viewCy - a.getY()) * GRAVITY_K;

            // 5. Verlet: update velocity then position
            a.setVx((a.getVx() + fx) * DAMPING);
            a.setVy((a.getVy() + fy) * DAMPING);

            // Clamp max velocity
            double speed = Math.sqrt(a.getVx() * a.getVx() + a.getVy() * a.getVy());
            if (speed > 20) {
                a.setVx(a.getVx() / speed * 20);
                a.setVy(a.getVy() / speed * 20);
            }

            double nx = a.getX() + a.getVx();
            double ny = a.getY() + a.getVy();

            // 6. Elastic wall bounce (in canvas coords)
            double wMin = WALL_MARGIN, wMax = getWidth() / zoom - WALL_MARGIN;
            double hMin = WALL_MARGIN, hMax = getHeight() / zoom - WALL_MARGIN;
            if (nx < wMin) {
                nx = wMin;
                a.setVx(-a.getVx() * WALL_BOUNCE);
            }
            if (nx > wMax) {
                nx = wMax;
                a.setVx(-a.getVx() * WALL_BOUNCE);
            }
            if (ny < hMin) {
                ny = hMin;
                a.setVy(-a.getVy() * WALL_BOUNCE);
            }
            if (ny > hMax) {
                ny = hMax;
                a.setVy(-a.getVy() * WALL_BOUNCE);
            }

            a.setX(nx);
            a.setY(ny);
        }
    }

    // =========================================================================
    // Rendering
    // =========================================================================
    public void render() {
        // FPS tracking
        long now = System.nanoTime();
        fpsCounter++;
        if (now - fpsTimer > 1_000_000_000L) {
            fps = fpsCounter;
            fpsCounter = 0;
            fpsTimer = now;
        }
        lastRenderNs = now;

        GraphicsContext gc = getGraphicsContext2D();
        double w = getWidth(), h = getHeight();

        // 1. Background — deep space gradient
        LinearGradient bg = new LinearGradient(0, 0, 0, 1, true, CycleMethod.NO_CYCLE,
                new Stop(0.0, Color.web("#060a14")),
                new Stop(1.0, Color.web("#0d1117")));
        gc.setFill(bg);
        gc.fillRect(0, 0, w, h);

        // 2. Grid dots
        drawGrid(gc, w, h);

        if (nodes == null || nodes.isEmpty()) {
            gc.setFill(Color.web("#30363d"));
            gc.setFont(Font.font("Segoe UI", 14));
            gc.setTextAlign(TextAlignment.CENTER);
            gc.fillText("Click 'Refresh Graph' to populate the knowledge graph", w / 2, h / 2);
            return;
        }

        gc.save();
        gc.translate(offsetX, offsetY);
        gc.scale(zoom, zoom);

        // 3. Edges (under nodes)
        for (KnowledgeEdge edge : edges)
            drawEdge(gc, edge);

        // 4. Nodes
        for (KnowledgeNode node : nodes)
            drawNode(gc, node);

        gc.restore();

        // 5. HUD overlay
        drawHUD(gc, w, h);
    }

    // ── Grid ─────────────────────────────────────────────────────────────────
    private void drawGrid(GraphicsContext gc, double w, double h) {
        gc.setStroke(Color.web("#161b22", 0.6));
        gc.setLineWidth(0.5);
        double gridSpacing = 40 * zoom;
        double startX = offsetX % gridSpacing;
        double startY = offsetY % gridSpacing;
        for (double x = startX; x < w; x += gridSpacing)
            gc.strokeLine(x, 0, x, h);
        for (double y = startY; y < h; y += gridSpacing)
            gc.strokeLine(0, y, w, y);
    }

    // ── Node ──────────────────────────────────────────────────────────────────
    private void drawNode(GraphicsContext gc, KnowledgeNode node) {
        double x = node.getX(), y = node.getY();
        Color color = getNodeColor(node.getType());
        boolean isSelected = (node == selectedNode);
        boolean isHovered = (node == hoveredNode);
        boolean isGlowing = node.isRecentlyActive(GLOW_WINDOW_MS);
        boolean isPinned = node.isPinned();

        // Glow halo
        if (isGlowing || isSelected || isHovered) {
            double glowAlpha = isGlowing
                    ? (node.isActive() ? 0.65
                            : Math.max(0,
                                    1.0 - (System.currentTimeMillis() - node.getLastActivatedMs())
                                            / (double) GLOW_WINDOW_MS)
                                    * 0.65)
                    : (isSelected ? 0.45 : 0.25);
            double glowR = isSelected ? GLOW_RADIUS * 1.15 : GLOW_RADIUS;
            gc.setFill(new RadialGradient(0, 0, x, y, glowR, false, CycleMethod.NO_CYCLE,
                    new Stop(0, color.deriveColor(0, 1, 1.4, glowAlpha)),
                    new Stop(0.5, color.deriveColor(0, 1, 1, glowAlpha * 0.4)),
                    new Stop(1, Color.TRANSPARENT)));
            gc.fillOval(x - glowR, y - glowR, glowR * 2, glowR * 2);
        }

        // Selection pulse ring
        if (isSelected) {
            long pulse = (System.currentTimeMillis() % 1200);
            double pulseR = NODE_RADIUS + 4 + (PULSE_MAX_R - NODE_RADIUS - 4) * (pulse / 1200.0);
            double pulseAlpha = 1.0 - pulse / 1200.0;
            gc.setStroke(color.deriveColor(0, 1, 1.5, pulseAlpha));
            gc.setLineWidth(2.0);
            gc.strokeOval(x - pulseR, y - pulseR, pulseR * 2, pulseR * 2);

            // Solid selection ring
            gc.setStroke(Color.WHITE.deriveColor(0, 1, 1, 0.9));
            gc.setLineWidth(2.5);
            gc.strokeOval(x - NODE_RADIUS - 5, y - NODE_RADIUS - 5,
                    (NODE_RADIUS + 5) * 2, (NODE_RADIUS + 5) * 2);
        }

        // Node body — radial gradient for 3D "sphere" look
        gc.setFill(new RadialGradient(
                -0.3, -0.3, x - NODE_RADIUS * 0.35, y - NODE_RADIUS * 0.35,
                NODE_RADIUS * 1.4, false, CycleMethod.NO_CYCLE,
                new Stop(0, color.brighter().brighter()),
                new Stop(0.5, color),
                new Stop(1, color.darker().darker())));
        gc.fillOval(x - NODE_RADIUS, y - NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2);

        // Node border
        gc.setStroke(isSelected ? Color.WHITE : color.brighter());
        gc.setLineWidth(isSelected ? 2.5 : 1.5);
        gc.setLineDashes(isPinned ? new double[] { 4, 3 } : null);
        gc.strokeOval(x - NODE_RADIUS, y - NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2);
        gc.setLineDashes(null);

        // Type icon
        gc.setFill(Color.WHITE);
        gc.setFont(Font.font("Segoe UI Emoji", FontWeight.BOLD, 13));
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText(getTypeIcon(node.getType()), x, y + 5);

        // Label below
        gc.setFill(isSelected ? Color.WHITE : Color.web("#c9d1d9"));
        gc.setFont(Font.font("Segoe UI", isSelected ? FontWeight.BOLD : FontWeight.NORMAL, zoom > 0.7 ? 11 : 9));
        String label = node.getLabel().length() > 22 ? node.getLabel().substring(0, 20) + ".." : node.getLabel();
        gc.fillText(label, x, y + NODE_RADIUS + 15);

        // Pin indicator
        if (isPinned) {
            gc.setFill(Color.web("#f0883e"));
            gc.setFont(Font.font("Segoe UI Emoji", 10));
            gc.fillText("📌", x + NODE_RADIUS - 2, y - NODE_RADIUS + 10);
        }

        // Mass badge (for hub nodes with degree ≥ 5)
        if (node.getMass() >= MASS_BASE + 5 * MASS_PER_EDGE) {
            gc.setFill(Color.web("#ffd700", 0.85));
            gc.setFont(Font.font("Segoe UI", 8));
            gc.fillText("HUB", x, y - NODE_RADIUS - 5);
        }
    }

    // ── Edge ──────────────────────────────────────────────────────────────────
    private void drawEdge(GraphicsContext gc, KnowledgeEdge edge) {
        KnowledgeNode src = findNode(edge.getSourceId());
        KnowledgeNode tgt = findNode(edge.getTargetId());
        if (src == null || tgt == null)
            return;

        boolean isSelected = (edge == selectedEdge);
        boolean isGlowing = edge.isRecentlyActive(GLOW_WINDOW_MS);
        Color edgeColor = getEdgeColor(edge.getRelationship());
        double alpha = isGlowing ? 1.0 : (isSelected ? 0.85 : 0.38);

        // Self-loop
        if (src.getId().equals(tgt.getId())) {
            drawSelfLoop(gc, src, edgeColor, alpha, isSelected || isGlowing);
            return;
        }

        double x1 = src.getX(), y1 = src.getY();
        double x2 = tgt.getX(), y2 = tgt.getY();
        double angle = Math.atan2(y2 - y1, x2 - x1);
        double sx = x1 + Math.cos(angle) * NODE_RADIUS;
        double sy = y1 + Math.sin(angle) * NODE_RADIUS;
        double ex = x2 - Math.cos(angle) * (NODE_RADIUS + 8);
        double ey = y2 - Math.sin(angle) * (NODE_RADIUS + 8);

        // Check for bidirectional pair — offset curve slightly
        boolean hasBidi = edges.stream()
                .anyMatch(e -> e != edge && e.getSourceId().equals(tgt.getId()) && e.getTargetId().equals(src.getId()));

        double cwx = 0, cwy = 0;
        if (hasBidi) {
            double perpX = -Math.sin(angle) * 30;
            double perpY = Math.cos(angle) * 30;
            double midX = (sx + ex) / 2 + perpX;
            double midY = (sy + ey) / 2 + perpY;
            gc.setStroke(edgeColor.deriveColor(0, 1, 1, alpha));
            gc.setLineWidth(isGlowing ? 3.0 : (isSelected ? 2.0 : 1.2));
            gc.setLineDashes(null);
            gc.beginPath();
            gc.moveTo(sx, sy);
            gc.quadraticCurveTo(midX, midY, ex, ey);
            gc.stroke();
            cwx = midX;
            cwy = midY;
        } else {
            gc.setStroke(edgeColor.deriveColor(0, 1, 1, alpha));
            gc.setLineWidth(isGlowing ? 3.0 : (isSelected ? 2.0 : 1.2));
            gc.strokeLine(sx, sy, ex, ey);
            cwx = (sx + ex) / 2;
            cwy = (sy + ey) / 2;
        }

        // Arrowhead
        double arrowLen = 10;
        double arrowAngle = Math.PI / 7;
        gc.setFill(edgeColor.deriveColor(0, 1, 1, alpha));
        gc.fillPolygon(
                new double[] { ex, ex - arrowLen * Math.cos(angle - arrowAngle),
                        ex - arrowLen * Math.cos(angle + arrowAngle) },
                new double[] { ey, ey - arrowLen * Math.sin(angle - arrowAngle),
                        ey - arrowLen * Math.sin(angle + arrowAngle) },
                3);

        // Edge label at midpoint (always shown when selected/glowing, on hover
        // otherwise)
        if (isSelected || isGlowing) {
            gc.setFill(Color.web("#f0f6fc", 0.9));
            gc.setFont(Font.font("Segoe UI", 9));
            gc.setTextAlign(TextAlignment.CENTER);
            gc.fillText(edge.getRelationship().name(), cwx, cwy - 6);
        }

        // Flow animation — animated dot along the edge when active
        if (isGlowing) {
            double t = (System.currentTimeMillis() % 1500) / 1500.0;
            double dotX = sx + (ex - sx) * t;
            double dotY = sy + (ey - sy) * t;
            gc.setFill(Color.WHITE.deriveColor(0, 1, 1, 0.9));
            gc.fillOval(dotX - 3.5, dotY - 3.5, 7, 7);
        }
    }

    private void drawSelfLoop(GraphicsContext gc, KnowledgeNode node, Color color, double alpha, boolean highlight) {
        double lx = node.getX() + NODE_RADIUS;
        double ly = node.getY() - NODE_RADIUS;
        gc.setStroke(color.deriveColor(0, 1, 1, alpha));
        gc.setLineWidth(highlight ? 2.5 : 1.2);
        gc.strokeOval(lx - 10, ly - 10, 30, 30);
    }

    // ── HUD ───────────────────────────────────────────────────────────────────
    private void drawHUD(GraphicsContext gc, double w, double h) {
        // Bottom-left info bar
        gc.setFill(Color.web("#161b22", 0.75));
        gc.fillRoundRect(6, h - 32, 360, 24, 6, 6);
        gc.setFill(Color.web("#8b949e"));
        gc.setFont(Font.font("Consolas", 11));
        gc.setTextAlign(TextAlignment.LEFT);
        gc.fillText(String.format("Nodes: %d  Edges: %d  Zoom: %.0f%%  FPS: %d",
                nodes.size(), edges.size(), zoom * 100, fps), 14, h - 15);

        // Legend top-right
        if (zoom > 0.5) {
            drawLegend(gc, w, h);
        }
    }

    private void drawLegend(GraphicsContext gc, double w, double h) {
        NodeType[] types = NodeType.values();
        double legendX = w - 160, legendY = 12;
        double rowH = 18;
        gc.setFill(Color.web("#161b22", 0.85));
        gc.fillRoundRect(legendX - 6, legendY - 2, 152, types.length * rowH + 8, 8, 8);
        gc.setFont(Font.font("Segoe UI", 10));
        for (int i = 0; i < types.length; i++) {
            double ry = legendY + i * rowH + rowH / 2;
            gc.setFill(getNodeColor(types[i]));
            gc.fillOval(legendX + 2, ry - 5, 10, 10);
            gc.setFill(Color.web("#c9d1d9"));
            gc.setTextAlign(TextAlignment.LEFT);
            gc.fillText(types[i].name().replace("_", " "), legendX + 18, ry + 4);
        }
    }

    // =========================================================================
    // Color / Icon maps
    // =========================================================================
    private Color getNodeColor(NodeType type) {
        return switch (type) {
            case ML_MODEL -> Color.web("#58a6ff");
            case RL_STATE -> Color.web("#3fb950");
            case LLM_LAYER -> Color.web("#bc8cff");
            case SLM_COMPONENT -> Color.web("#d2a8ff");
            case GENAI_MODULE -> Color.web("#f778ba");
            case KNOWLEDGE_INSIGHT -> Color.web("#d29922");
            case DATA_SOURCE -> Color.web("#8b949e");
            case PIPELINE_STAGE -> Color.web("#79c0ff");
            case ONNX_EXPORT -> Color.web("#56d364");
            case CONSTITUTIONAL_RULE -> Color.web("#f85149");
            case LCA_LAYER -> Color.web("#f0883e"); // Amber
            case DL_BACKBONE -> Color.web("#76e3ea"); // Cyan
            case FINE_TUNE_JOB -> Color.web("#c9d1d9"); // Pearl
        };
    }

    private String getTypeIcon(NodeType type) {
        return switch (type) {
            case ML_MODEL -> "M";
            case RL_STATE -> "R";
            case LLM_LAYER -> "L";
            case SLM_COMPONENT -> "S";
            case GENAI_MODULE -> "G";
            case KNOWLEDGE_INSIGHT -> "K";
            case DATA_SOURCE -> "D";
            case PIPELINE_STAGE -> "P";
            case ONNX_EXPORT -> "O";
            case CONSTITUTIONAL_RULE -> "C";
            case LCA_LAYER -> "A"; // Agent
            case DL_BACKBONE -> "N"; // Neural
            case FINE_TUNE_JOB -> "F"; // Fine-tune
        };
    }

    private Color getEdgeColor(Relationship rel) {
        return switch (rel) {
            case TRAINS_ON -> Color.web("#58a6ff");
            case FEEDS_INTO -> Color.web("#79c0ff");
            case TRIGGERS -> Color.web("#f0883e");
            case LEARNED_FROM -> Color.web("#3fb950");
            case FINE_TUNES -> Color.web("#bc8cff");
            case TRANSITIONS_TO -> Color.web("#56d364");
            case EXPORTS_TO -> Color.web("#8b949e");
            case GUARDS -> Color.web("#f85149");
            case RECOMMENDS -> Color.web("#d29922");
            case OBSERVES -> Color.web("#d2a8ff");
        };
    }

    // =========================================================================
    // Hit testing
    // =========================================================================
    private KnowledgeNode hitTestNode(double mx, double my) {
        for (KnowledgeNode node : nodes) {
            double dx = node.getX() - mx, dy = node.getY() - my;
            if (Math.sqrt(dx * dx + dy * dy) < NODE_RADIUS + 4)
                return node;
        }
        return null;
    }

    private KnowledgeEdge hitTestEdge(double mx, double my) {
        for (KnowledgeEdge edge : edges) {
            KnowledgeNode src = findNode(edge.getSourceId());
            KnowledgeNode tgt = findNode(edge.getTargetId());
            if (src == null || tgt == null || src == tgt)
                continue;
            if (pointToSegmentDist(mx, my, src.getX(), src.getY(), tgt.getX(), tgt.getY()) < 9)
                return edge;
        }
        return null;
    }

    // =========================================================================
    // Mouse handlers
    // =========================================================================
    private void handleMousePressed(MouseEvent e) {
        double mx = (e.getX() - offsetX) / zoom;
        double my = (e.getY() - offsetY) / zoom;

        KnowledgeNode hitNode = hitTestNode(mx, my);
        if (hitNode != null) {
            if (e.getButton() == MouseButton.PRIMARY) {
                if (e.getClickCount() == 2) {
                    hitNode.setPinned(!hitNode.isPinned()); // Double-click to pin/unpin
                    render();
                    return;
                }
                draggedNode = hitNode;
                selectedNode = hitNode;
                selectedEdge = null;
                if (onNodeSelected != null) {
                    List<KnowledgeEdge> connected = edges.stream()
                            .filter(edge -> edge.getSourceId().equals(hitNode.getId())
                                    || edge.getTargetId().equals(hitNode.getId()))
                            .toList();
                    onNodeSelected.accept(hitNode, connected);
                }
            }
            render();
            return;
        }

        KnowledgeEdge hitEdge = hitTestEdge(mx, my);
        if (hitEdge != null) {
            selectedEdge = hitEdge;
            selectedNode = null;
            KnowledgeNode src = findNode(hitEdge.getSourceId());
            KnowledgeNode tgt = findNode(hitEdge.getTargetId());
            if (onEdgeSelected != null && src != null && tgt != null)
                onEdgeSelected.accept(hitEdge, new KnowledgeNode[] { src, tgt });
            render();
            return;
        }

        // Pan canvas
        selectedNode = null;
        selectedEdge = null;
        isDraggingCanvas = true;
        lastMouseX = e.getX();
        lastMouseY = e.getY();
        render();
    }

    private void handleMouseDragged(MouseEvent e) {
        if (draggedNode != null) {
            draggedNode.setX((e.getX() - offsetX) / zoom);
            draggedNode.setY((e.getY() - offsetY) / zoom);
            draggedNode.setVx(0);
            draggedNode.setVy(0);
        } else if (isDraggingCanvas) {
            offsetX += e.getX() - lastMouseX;
            offsetY += e.getY() - lastMouseY;
            lastMouseX = e.getX();
            lastMouseY = e.getY();
        }
        render();
    }

    private void handleMouseReleased(MouseEvent e) {
        draggedNode = null;
        isDraggingCanvas = false;
    }

    private void handleMouseMoved(MouseEvent e) {
        double mx = (e.getX() - offsetX) / zoom;
        double my = (e.getY() - offsetY) / zoom;
        KnowledgeNode hit = hitTestNode(mx, my);
        if (hit != hoveredNode) {
            hoveredNode = hit;
            render();
        }
    }

    private void handleScroll(ScrollEvent e) {
        double factor = e.getDeltaY() > 0 ? 1.12 : (1 / 1.12);
        double newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * factor));
        // Zoom toward mouse cursor
        double mx = e.getX(), my = e.getY();
        offsetX = mx - (mx - offsetX) * (newZoom / zoom);
        offsetY = my - (my - offsetY) * (newZoom / zoom);
        zoom = newZoom;
        render();
    }

    // =========================================================================
    // Utilities
    // =========================================================================
    private KnowledgeNode findNode(String id) {
        if (nodes == null || id == null)
            return null;
        for (KnowledgeNode n : nodes)
            if (n.getId().equals(id))
                return n;
        return null;
    }

    private double pointToSegmentDist(double px, double py, double ax, double ay, double bx, double by) {
        double dx = bx - ax, dy = by - ay;
        double lenSq = dx * dx + dy * dy;
        if (lenSq == 0)
            return Math.sqrt((px - ax) * (px - ax) + (py - ay) * (py - ay));
        double t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
        double projX = ax + t * dx, projY = ay + t * dy;
        return Math.sqrt((px - projX) * (px - projX) + (py - projY) * (py - projY));
    }
}
