const SVG_NS = "http://www.w3.org/2000/svg";

function setMathNodeVisibility(mathNode, isVisible) {
    const opacity = isVisible ? "1" : "0";
    const visibility = isVisible ? "visible" : "hidden";

    mathNode.fo.style.opacity = opacity;
    mathNode.fo.style.visibility = visibility;
    mathNode.div.style.opacity = opacity;
    mathNode.div.style.visibility = visibility;
}

function escapeHtml(text) {
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

const CODE_SNIPPETS = {
    ba: {
        lines: [
            { number: 1, text: "# each observation picks one camera and one landmark" },
            { number: 2, text: "input = {" },
            { number: 3, text: "    \"camera_indices\": trimmed_dataset['camera_index_of_observations']," },
            { number: 4, text: "    \"point_indices\": trimmed_dataset['point_index_of_observations']," },
            { number: 5, text: "}" },
            { number: 6, text: "model = Reproj(camera_params, points_3d)" },
            { number: 7, text: "residual = model(points_2d, input[\"camera_indices\"], input[\"point_indices\"])" },
            { number: 8, text: "loss = optimizer.step(input)  # builds the sparse Jacobian" },
        ],
        phases: {
            idle: {
                label: "Ready",
                caption: "A distilled BA code path appears here so readers can match the animation to the idea in a few seconds.",
                lines: [],
            },
            inputs: {
                label: "Inputs",
                caption: "Observation indices decide which camera block and which landmark block each residual will touch.",
                lines: [2, 3, 4, 5],
            },
            replication: {
                label: "Replication",
                caption: "The replication stage is just those index arrays being used over and over across observations.",
                lines: [3, 4],
            },
            residuals: {
                label: "Residuals",
                caption: "The selected camera-landmark pair produces one reprojection residual.",
                lines: [6, 7],
            },
            jacobian: {
                label: "Jacobian",
                caption: "Inside the optimizer step, that residual graph is differentiated into sparse Jacobian blocks.",
                lines: [8],
            },
            complete: {
                label: "Solve",
                caption: "Once the Jacobian is built, the solver step runs and the pattern repeats on the next iteration.",
                lines: [8],
            },
        },
    },
    gauge: {
        lines: [
            { number: 1, text: "# fix the first camera so its Jacobian block disappears" },
            { number: 2, text: "camera_fixed = camera_se3[:1].clone()" },
            { number: 3, text: "camera_se3 = torch.cat([camera_fixed, self.pose_rest], dim=0)" },
            { number: 4, text: "residual = project_with_se3_and_intrinsics(" },
            { number: 5, text: "    self.points_3d[point_indices], camera_se3[camera_indices], self.intrinsics[camera_indices]" },
            { number: 6, text: ") - points_2d" },
            { number: 7, text: "J_cam_rest, J_intr, J_pts = autograd_graph.jacobian(" },
            { number: 8, text: "    residual, [model.pose_rest, model.intrinsics, model.points_3d]" },
            { number: 9, text: ")" },
            { number: 10, text: "expected_cam_cols = camera_idx[camera_idx > 0] - 1" },
        ],
        phases: {
            idle: {
                label: "Ready",
                caption: "This version strips the gauge-fixing story down to the few lines that matter for the Jacobian structure.",
                lines: [],
            },
            inputs: {
                label: "Fixed Gauge",
                caption: "The first camera is split out and held fixed before optimization starts.",
                lines: [1, 2],
            },
            replication: {
                label: "Replication",
                caption: "The fixed camera is concatenated back only for lookup; the trainable camera state is still just pose_rest.",
                lines: [3],
            },
            residuals: {
                label: "Residuals",
                caption: "Residuals still come from the selected point, selected camera pose, and selected intrinsics.",
                lines: [4, 5, 6],
            },
            jacobian: {
                label: "Jacobian",
                caption: "The Jacobian is taken only with respect to the unfixed camera poses, intrinsics, and landmarks.",
                lines: [7, 8, 9],
            },
            complete: {
                label: "Gauge-Free",
                caption: "That is why the camera columns are reindexed to skip camera 0.",
                lines: [10],
            },
        },
    },
    pgo: {
        lines: [
            { number: 1, text: "@map_transform" },
            { number: 2, text: "def _tracked_pose_graph_residual(poses, node1, node2, infos):" },
            { number: 3, text: "    residual = (pp.SE3(poses).Inv() @ pp.SE3(node1).Inv() @ pp.SE3(node2)).Log().tensor()" },
            { number: 4, text: "    return (infos @ residual[..., None])[..., 0]" },
            { number: 5, text: "input = {\"edges\": edges, \"poses\": poses, \"infos\": infos}" },
            { number: 6, text: "node1 = self.nodes[edges[..., 0]]" },
            { number: 7, text: "node2 = self.nodes[edges[..., 1]]" },
            { number: 8, text: "residual = _tracked_pose_graph_residual(poses, node1, node2, infos)" },
            { number: 9, text: "loss = optimizer.step(input=input, weight=infos)" },
        ],
        phases: {
            idle: {
                label: "Ready",
                caption: "The PGO tab now shows only the core residual path, so the compute graph reads almost left-to-right with the animation.",
                lines: [],
            },
            inputs: {
                label: "Inputs",
                caption: "The input is just edge connectivity, relative poses, and information weights.",
                lines: [5],
            },
            replication: {
                label: "Replication",
                caption: "Each edge selects the two pose blocks it connects.",
                lines: [6, 7],
            },
            residuals: {
                label: "Residuals",
                caption: "Those two selected poses and the measurement produce one weighted SE(3) residual.",
                lines: [1, 2, 3, 4, 8],
            },
            jacobian: {
                label: "Jacobian",
                caption: "The optimizer differentiates that residual graph into sparse Jacobian blocks.",
                lines: [9],
            },
            complete: {
                label: "Solve",
                caption: "Then the sparse solve runs and the same pattern repeats for the next iteration.",
                lines: [9],
            },
        },
    },
};

class CodePanel {
    constructor(sceneKey) {
        this.sceneKey = sceneKey;
        this.definition = CODE_SNIPPETS[sceneKey];
        this.block = document.getElementById(`${sceneKey}-code-block`);
        this.caption = document.getElementById(`${sceneKey}-code-caption`);
        this.stepPill = document.getElementById(`${sceneKey}-step-pill`);
        this.lineNodes = new Map();
        this.activePhase = "idle";
        this.render();
        this.setPhase("idle");
    }

    render() {
        const html = this.definition.lines.map(({ number, text }) => {
            return [
                `<div class="code-line" data-line="${number}">`,
                `<span class="code-line-no">${number}</span>`,
                `<span class="code-line-text">${escapeHtml(text)}</span>`,
                "</div>",
            ].join("");
        }).join("");

        this.block.innerHTML = html;
        this.lineNodes = new Map(
            this.definition.lines.map(({ number }) => [
                number,
                this.block.querySelector(`.code-line[data-line="${number}"]`),
            ]),
        );
    }

    setPhase(phaseKey) {
        const phase = this.definition.phases[phaseKey] || this.definition.phases.idle;
        const activeLines = new Set(phase.lines);
        this.activePhase = phaseKey;

        this.stepPill.textContent = phase.label;
        this.caption.textContent = phase.caption;
        this.block.classList.toggle("has-active", activeLines.size > 0);

        let firstActiveNode = null;
        this.lineNodes.forEach((node, lineNumber) => {
            const isActive = activeLines.has(lineNumber);
            node.classList.toggle("active", isActive);
            if (isActive && firstActiveNode === null) {
                firstActiveNode = node;
            }
        });

        if (firstActiveNode) {
            firstActiveNode.scrollIntoView({ block: "nearest" });
        } else {
            this.block.scrollTop = 0;
        }
    }

    reset() {
        this.setPhase("idle");
    }
}

class BaseCanvas {
    constructor(containerId, sceneKey) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.svg = document.createElementNS(SVG_NS, "svg");
        this.svg.setAttribute("viewBox", "0 0 1000 600");
        this.container.appendChild(this.svg);
        this.elements = {};
        this.timeouts = [];
        this.codePanel = new CodePanel(sceneKey);
    }

    addMath(content, x, y, width, height, classes) {
        const foreignObject = document.createElementNS(SVG_NS, "foreignObject");
        foreignObject.setAttribute("x", x);
        foreignObject.setAttribute("y", y);
        foreignObject.setAttribute("width", width);
        foreignObject.setAttribute("height", height);
        foreignObject.style.pointerEvents = "none";
        foreignObject.style.transition = "opacity 0.3s ease";
        foreignObject.style.visibility = "visible";

        const div = document.createElement("div");
        div.style.width = "100%";
        div.style.height = "100%";
        div.style.display = "flex";
        div.style.justifyContent = "center";
        div.style.alignItems = "center";
        div.style.opacity = "1";
        div.style.visibility = "visible";
        if (classes) {
            div.className = classes;
        }

        katex.render(content, div, { throwOnError: false, displayMode: true });

        foreignObject.appendChild(div);
        this.mainGroup.appendChild(foreignObject);
        return { fo: foreignObject, div };
    }

    addRect(x, y, width, height, classes) {
        const rect = document.createElementNS(SVG_NS, "rect");
        rect.setAttribute("x", x);
        rect.setAttribute("y", y);
        rect.setAttribute("width", width);
        rect.setAttribute("height", height);
        rect.setAttribute("class", `node-rect ${classes}`);
        this.mainGroup.appendChild(rect);
        return rect;
    }

    addText(content, x, y, classes) {
        const text = document.createElementNS(SVG_NS, "text");
        text.setAttribute("x", x);
        text.setAttribute("y", y);
        text.setAttribute("class", classes);
        text.textContent = content;
        this.mainGroup.appendChild(text);
        return text;
    }

    addPath(x1, y1, x2, y2, classes) {
        const path = document.createElementNS(SVG_NS, "path");
        const dx = Math.abs(x2 - x1) * 0.5;
        const d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
        path.setAttribute("d", d);
        path.setAttribute("class", classes);
        this.mainGroup.appendChild(path);

        if (classes.includes("edge-flow")) {
            const length = path.getTotalLength();
            path.style.strokeDasharray = length;
            path.style.strokeDashoffset = length;
        }
        return path;
    }

    reset() {
        this.timeouts.forEach((timeoutId) => clearTimeout(timeoutId));
        this.timeouts = [];
        this.build();
        this.codePanel.reset();
    }
}

class BACanvas extends BaseCanvas {
    constructor(containerId, isGaugeFixed, sceneKey) {
        super(containerId, sceneKey);
        this.isGaugeFixed = isGaugeFixed;
        this.build();
        this.codePanel.reset();
    }

    build() {
        while (this.svg.firstChild) {
            this.svg.removeChild(this.svg.firstChild);
        }

        this.elements = {
            edges: [],
            repZ: [],
            repP: [],
            res: [],
            jacZEdges: [],
            jacPEdges: [],
            jacZBlocks: [],
            jacPBlocks: [],
            staticEdges: [],
        };

        const group = document.createElementNS(SVG_NS, "g");
        this.svg.appendChild(group);
        this.mainGroup = group;

        const numZ = 2;
        const numP = 3;
        const observations = [
            { c: 0, p: 0 },
            { c: 0, p: 1 },
            { c: 1, p: 1 },
            { c: 1, p: 2 },
        ];

        const zX = 100;
        const zStartY = 150;
        const zGap = 60;
        const pX = 100;
        const pStartY = 350;
        const pGap = 60;
        const repZX = 350;
        const repPX = 430;
        const resX = 550;
        const repStartY = 200;
        const resGap = 60;
        const jzX = 700;
        const jpX = 840;
        const jStartY = 200;
        const blockSize = 55;
        const blockGap = 6;

        this.addText("Z (Cameras)", zX + 20, zStartY - 40, "title-label");
        this.addText("P (Landmarks)", pX + 20, pStartY - 40, "title-label");
        this.addText("Replication", (repZX + repPX) / 2 + 20, repStartY - 40, "title-label");
        this.addText("Residuals R", resX + 20, repStartY - 40, "title-label");
        this.addMath("\\frac{\\partial \\mathbf{r}}{\\partial \\mathbf{Z}}", jzX, jStartY - 60, numZ * (blockSize + blockGap) - blockGap, 40, "title-label");
        this.addMath("\\frac{\\partial \\mathbf{r}}{\\partial \\mathbf{P}}", jpX, jStartY - 60, numP * (blockSize + blockGap) - blockGap, 40, "title-label");

        const zNodes = [];
        for (let index = 0; index < numZ; index += 1) {
            const isFixed = this.isGaugeFixed && index === 0;
            const node = this.addRect(zX, zStartY + index * zGap, 40, 40, `cam-node ${isFixed ? "fixed-node" : ""}`);
            this.addText(`C${index}`, zX + 20, zStartY + index * zGap + 20, "label");
            zNodes.push({ x: zX + 40, y: zStartY + index * zGap + 20, el: node });
        }

        const pNodes = [];
        for (let index = 0; index < numP; index += 1) {
            const node = this.addRect(pX, pStartY + index * pGap, 40, 40, "pt-node");
            this.addText(`P${index}`, pX + 20, pStartY + index * pGap + 20, "label");
            pNodes.push({ x: pX + 40, y: pStartY + index * pGap + 20, el: node });
        }

        observations.forEach((observation, rowIndex) => {
            const y = repStartY + rowIndex * resGap;
            const edgeZFlow = this.addPath(zNodes[observation.c].x, zNodes[observation.c].y, repZX, y + 20, "edge-flow");
            const edgePFlow = this.addPath(pNodes[observation.p].x, pNodes[observation.p].y, repPX, y + 20, "edge-flow");
            this.elements.edges.push(edgeZFlow, edgePFlow);

            const staticEdge1 = this.addPath(repZX + 40, y + 20, resX, y + 20, "edge");
            const staticEdge2 = this.addPath(repPX + 40, y + 20, resX, y + 20, "edge");
            staticEdge1.style.opacity = "0";
            staticEdge2.style.opacity = "0";
            staticEdge1.style.transition = "opacity 0.5s ease-in-out";
            staticEdge2.style.transition = "opacity 0.5s ease-in-out";
            this.elements.staticEdges.push(staticEdge1, staticEdge2);

            const isFixed = this.isGaugeFixed && observation.c === 0;
            const repZNode = this.addRect(repZX, y, 40, 40, `cam-node ${isFixed ? "fixed-node" : ""}`);
            repZNode.style.opacity = "0";
            const repZText = this.addText(`C${observation.c}`, repZX + 20, y + 20, "label");
            repZText.style.opacity = "0";
            this.elements.repZ.push({ rect: repZNode, text: repZText });

            const repPNode = this.addRect(repPX, y, 40, 40, "pt-node");
            repPNode.style.opacity = "0";
            const repPText = this.addText(`P${observation.p}`, repPX + 20, y + 20, "label");
            repPText.style.opacity = "0";
            this.elements.repP.push({ rect: repPNode, text: repPText });

            const resNode = this.addRect(resX, y, 40, 40, "res-node");
            resNode.style.opacity = "0";
            const resText = this.addText(`r${rowIndex}`, resX + 20, y + 20, "label");
            resText.style.opacity = "0";
            this.elements.res.push({ rect: resNode, text: resText, y: y + 20 });
        });

        for (let row = 0; row < observations.length; row += 1) {
            for (let column = 0; column < numZ; column += 1) {
                const isNonZero = observations[row].c === column;
                const isFixed = this.isGaugeFixed && column === 0;
                const opacity = isNonZero && !isFixed ? 1 : 0.1;
                const bx = jzX + column * (blockSize + blockGap);
                const by = jStartY + row * (blockSize + blockGap);
                const block = this.addRect(bx, by, blockSize, blockSize, "cam-jac-node node-rect");
                block.style.opacity = "0";
                let mathNode = null;
                if (isNonZero && !isFixed) {
                    mathNode = this.addMath(`\\frac{\\partial r_{${row}}}{\\partial C_{${column}}}`, bx, by, blockSize, blockSize, "math-label");
                    setMathNodeVisibility(mathNode, false);
                }
                this.elements.jacZBlocks.push({ block, opacity, mathNode });
                if (isNonZero && !isFixed) {
                    const edge = this.addPath(resX + 40, this.elements.res[row].y, bx, by + blockSize / 2, "edge-flow");
                    this.elements.jacZEdges.push(edge);
                }
            }
        }

        for (let row = 0; row < observations.length; row += 1) {
            for (let column = 0; column < numP; column += 1) {
                const isNonZero = observations[row].p === column;
                const opacity = isNonZero ? 1 : 0.1;
                const bx = jpX + column * (blockSize + blockGap);
                const by = jStartY + row * (blockSize + blockGap);
                const block = this.addRect(bx, by, blockSize, blockSize, "pt-jac-node node-rect");
                block.style.opacity = "0";
                let mathNode = null;
                if (isNonZero) {
                    mathNode = this.addMath(`\\frac{\\partial r_{${row}}}{\\partial P_{${column}}}`, bx, by, blockSize, blockSize, "math-label");
                    setMathNodeVisibility(mathNode, false);
                }
                this.elements.jacPBlocks.push({ block, opacity, mathNode });
                if (isNonZero) {
                    const edge = this.addPath(resX + 40, this.elements.res[row].y, bx, by + blockSize / 2, "edge-flow");
                    this.elements.jacPEdges.push(edge);
                }
            }
        }
    }

    play() {
        this.reset();
        let elapsed = 0;
        const queueStep = (phase, delay, callback) => {
            elapsed += delay;
            this.timeouts.push(setTimeout(() => {
                this.codePanel.setPhase(phase);
                callback();
            }, elapsed));
        };

        queueStep("inputs", 100, () => {
            this.elements.edges.forEach((edge) => {
                edge.style.opacity = "0.8";
                edge.style.transition = "stroke-dashoffset 1s ease-in-out";
                edge.style.strokeDashoffset = "0";
            });
        });

        queueStep("replication", 1200, () => {
            this.elements.repZ.forEach((node) => {
                node.rect.style.opacity = node.rect.classList.contains("fixed-node") ? "0.4" : "1";
                node.text.style.opacity = "1";
            });
            this.elements.repP.forEach((node) => {
                node.rect.style.opacity = "1";
                node.text.style.opacity = "1";
            });
            this.elements.staticEdges.forEach((edge) => {
                edge.style.opacity = "0.6";
            });
            this.elements.edges.forEach((edge) => {
                edge.style.opacity = "0";
            });
        });

        queueStep("residuals", 800, () => {
            this.elements.res.forEach((node) => {
                node.rect.style.opacity = "1";
                node.text.style.opacity = "1";
            });
        });

        queueStep("jacobian", 800, () => {
            [...this.elements.jacZEdges, ...this.elements.jacPEdges].forEach((edge) => {
                edge.style.opacity = "0.8";
                edge.style.transition = "stroke-dashoffset 1s ease-in-out";
                edge.style.strokeDashoffset = "0";
            });
        });

        queueStep("complete", 1200, () => {
            this.elements.jacZBlocks.forEach((entry) => {
                entry.block.style.opacity = entry.opacity;
                if (entry.mathNode) {
                    setMathNodeVisibility(entry.mathNode, true);
                }
            });
            this.elements.jacPBlocks.forEach((entry) => {
                entry.block.style.opacity = entry.opacity;
                if (entry.mathNode) {
                    setMathNodeVisibility(entry.mathNode, true);
                }
            });
        });

        queueStep("complete", 800, () => {
            [...this.elements.jacZEdges, ...this.elements.jacPEdges].forEach((edge) => {
                edge.style.opacity = "0";
            });
        });
    }
}

class PGOCanvas extends BaseCanvas {
    constructor(containerId, sceneKey) {
        super(containerId, sceneKey);
        this.build();
        this.codePanel.reset();
    }

    build() {
        while (this.svg.firstChild) {
            this.svg.removeChild(this.svg.firstChild);
        }

        this.elements = {
            edges: [],
            repNodes: [],
            res: [],
            jacEdges: [],
            jacBlocks: [],
            staticEdges: [],
        };

        const group = document.createElementNS(SVG_NS, "g");
        this.svg.appendChild(group);
        this.mainGroup = group;

        const numNodes = 3;
        const edges = [
            { from: 0, to: 1 },
            { from: 1, to: 2 },
            { from: 0, to: 2 },
        ];

        const nX = 150;
        const nStartY = 200;
        const nGap = 80;
        const repX1 = 350;
        const repX2 = 450;
        const resX = 580;
        const rStartY = 200;
        const rGap = 80;
        const jX = 740;
        const jStartY = 200;
        const blockSize = 60;
        const blockGap = 6;

        this.addText("Poses X", nX + 20, nStartY - 40, "title-label");
        this.addText("Replication", (repX1 + repX2) / 2 + 20, rStartY - 40, "title-label");
        this.addText("Residuals R", resX + 20, rStartY - 40, "title-label");
        this.addMath("\\frac{\\partial \\mathbf{r}}{\\partial \\mathbf{X}}", jX, jStartY - 60, numNodes * (blockSize + blockGap) - blockGap, 40, "title-label");

        const nodes = [];
        for (let index = 0; index < numNodes; index += 1) {
            const node = this.addRect(nX, nStartY + index * nGap, 40, 40, "cam-node");
            this.addText(`X${index}`, nX + 20, nStartY + index * nGap + 20, "label");
            nodes.push({ x: nX + 40, y: nStartY + index * nGap + 20, el: node });
        }

        edges.forEach((edgeDef, rowIndex) => {
            const y = rStartY + rowIndex * rGap;
            const edgeFlow1 = this.addPath(nodes[edgeDef.from].x, nodes[edgeDef.from].y, repX1, y + 20, "edge-flow");
            const edgeFlow2 = this.addPath(nodes[edgeDef.to].x, nodes[edgeDef.to].y, repX2, y + 20, "edge-flow");
            this.elements.edges.push(edgeFlow1, edgeFlow2);

            const staticEdge1 = this.addPath(repX1 + 40, y + 20, resX, y + 20, "edge");
            const staticEdge2 = this.addPath(repX2 + 40, y + 20, resX, y + 20, "edge");
            staticEdge1.style.opacity = "0";
            staticEdge2.style.opacity = "0";
            staticEdge1.style.transition = "opacity 0.5s ease-in-out";
            staticEdge2.style.transition = "opacity 0.5s ease-in-out";
            this.elements.staticEdges.push(staticEdge1, staticEdge2);

            const rep1Node = this.addRect(repX1, y, 40, 40, "cam-node");
            rep1Node.style.opacity = "0";
            const rep1Text = this.addText(`X${edgeDef.from}`, repX1 + 20, y + 20, "label");
            rep1Text.style.opacity = "0";

            const rep2Node = this.addRect(repX2, y, 40, 40, "cam-node");
            rep2Node.style.opacity = "0";
            const rep2Text = this.addText(`X${edgeDef.to}`, repX2 + 20, y + 20, "label");
            rep2Text.style.opacity = "0";

            this.elements.repNodes.push({ rect: rep1Node, text: rep1Text }, { rect: rep2Node, text: rep2Text });

            const resNode = this.addRect(resX, y, 40, 40, "res-node");
            resNode.style.opacity = "0";
            const resText = this.addText(`r${rowIndex}`, resX + 20, y + 20, "label");
            resText.style.opacity = "0";
            this.elements.res.push({ rect: resNode, text: resText, y: y + 20 });
        });

        for (let row = 0; row < edges.length; row += 1) {
            for (let column = 0; column < numNodes; column += 1) {
                const isNonZero = edges[row].from === column || edges[row].to === column;
                const opacity = isNonZero ? 1 : 0.1;
                const bx = jX + column * (blockSize + blockGap);
                const by = jStartY + row * (blockSize + blockGap);
                const block = this.addRect(bx, by, blockSize, blockSize, "cam-jac-node node-rect");
                block.style.opacity = "0";
                let mathNode = null;
                if (isNonZero) {
                    mathNode = this.addMath(`\\frac{\\partial r_{${row}}}{\\partial X_{${column}}}`, bx, by, blockSize, blockSize, "math-label");
                    setMathNodeVisibility(mathNode, false);
                }
                this.elements.jacBlocks.push({ block, opacity, mathNode });
                if (isNonZero) {
                    const edge = this.addPath(resX + 40, this.elements.res[row].y, bx, by + blockSize / 2, "edge-flow");
                    this.elements.jacEdges.push(edge);
                }
            }
        }
    }

    play() {
        this.reset();
        let elapsed = 0;
        const queueStep = (phase, delay, callback) => {
            elapsed += delay;
            this.timeouts.push(setTimeout(() => {
                this.codePanel.setPhase(phase);
                callback();
            }, elapsed));
        };

        queueStep("inputs", 100, () => {
            this.elements.edges.forEach((edge) => {
                edge.style.opacity = "0.8";
                edge.style.transition = "stroke-dashoffset 1s ease-in-out";
                edge.style.strokeDashoffset = "0";
            });
        });

        queueStep("replication", 1200, () => {
            this.elements.repNodes.forEach((node) => {
                node.rect.style.opacity = "1";
                node.text.style.opacity = "1";
            });
            this.elements.staticEdges.forEach((edge) => {
                edge.style.opacity = "0.6";
            });
            this.elements.edges.forEach((edge) => {
                edge.style.opacity = "0";
            });
        });

        queueStep("residuals", 800, () => {
            this.elements.res.forEach((node) => {
                node.rect.style.opacity = "1";
                node.text.style.opacity = "1";
            });
        });

        queueStep("jacobian", 800, () => {
            this.elements.jacEdges.forEach((edge) => {
                edge.style.opacity = "0.8";
                edge.style.transition = "stroke-dashoffset 1s ease-in-out";
                edge.style.strokeDashoffset = "0";
            });
        });

        queueStep("complete", 1200, () => {
            this.elements.jacBlocks.forEach((entry) => {
                entry.block.style.opacity = entry.opacity;
                if (entry.mathNode) {
                    setMathNodeVisibility(entry.mathNode, true);
                }
            });
        });

        queueStep("complete", 800, () => {
            this.elements.jacEdges.forEach((edge) => {
                edge.style.opacity = "0";
            });
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabButtons.forEach((button) => {
        button.addEventListener("click", () => {
            tabButtons.forEach((candidate) => {
                candidate.classList.remove("active");
            });
            tabContents.forEach((content) => {
                content.classList.remove("active");
            });
            button.classList.add("active");
            const targetId = button.getAttribute("data-target");
            document.getElementById(targetId).classList.add("active");
        });
    });

    const baScene = new BACanvas("ba-viz", false, "ba");
    const gaugeScene = new BACanvas("gauge-viz", true, "gauge");
    const pgoScene = new PGOCanvas("pgo-viz", "pgo");

    document.getElementById("play-ba").addEventListener("click", () => baScene.play());
    document.getElementById("reset-ba").addEventListener("click", () => baScene.reset());

    document.getElementById("play-gauge").addEventListener("click", () => gaugeScene.play());
    document.getElementById("reset-gauge").addEventListener("click", () => gaugeScene.reset());

    document.getElementById("play-pgo").addEventListener("click", () => pgoScene.play());
    document.getElementById("reset-pgo").addEventListener("click", () => pgoScene.reset());
});
