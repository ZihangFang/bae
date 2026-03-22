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

function splitInlineComment(line) {
    let inSingleQuote = false;
    let inDoubleQuote = false;
    let escaped = false;

    for (let index = 0; index < line.length; index += 1) {
        const char = line[index];

        if (escaped) {
            escaped = false;
            continue;
        }

        if (char === "\\") {
            escaped = true;
            continue;
        }

        if (char === "'" && !inDoubleQuote) {
            inSingleQuote = !inSingleQuote;
            continue;
        }

        if (char === "\"" && !inSingleQuote) {
            inDoubleQuote = !inDoubleQuote;
            continue;
        }

        if (char === "#" && !inSingleQuote && !inDoubleQuote) {
            return [line.slice(0, index), line.slice(index)];
        }
    }

    return [line, ""];
}

function highlightPythonCode(code) {
    const strings = [];
    const withPlaceholders = code.replace(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, (match) => {
        const placeholder = `__PY_STRING_${strings.length}__`;
        strings.push(match);
        return placeholder;
    });

    let html = escapeHtml(withPlaceholders);
    html = html.replace(/(^|\s)(@[A-Za-z_][A-Za-z0-9_]*)/g, (match, prefix, decorator) => {
        return `${prefix}<span class="token-decorator">${decorator}</span>`;
    });
    html = html.replace(/\b(def|for|in|return)\b/g, `<span class="token-keyword">$1</span>`);
    html = html.replace(/\b([A-Za-z_][A-Za-z0-9_]*)\b(?=\()/g, `<span class="token-function">$1</span>`);

    strings.forEach((value, index) => {
        html = html.replace(`__PY_STRING_${index}__`, `<span class="token-string">${escapeHtml(value)}</span>`);
    });

    return html;
}

function renderPythonLine(line, lineNumber) {
    const [codePart, commentPart] = splitInlineComment(line);
    let codeHtml = highlightPythonCode(codePart);
    if (commentPart) {
        codeHtml += `<span class="token-comment">${escapeHtml(commentPart)}</span>`;
    }
    if (!codeHtml) {
        codeHtml = "&nbsp;";
    }

    return [
        `<span class="phase-card-line">`,
        `<span class="phase-card-line-no">${lineNumber}</span>`,
        `<span class="phase-card-line-code">${codeHtml}</span>`,
        `</span>`,
    ].join("");
}

const CODE_SNIPPETS = {
    ba: {
        intro: {
            label: "Ready",
            caption: "Each card matches one stage in the animation. Click any card to jump straight to that step.",
        },
        phases: [
            {
                key: "inputs",
                label: "Inputs",
                caption: "Observation indices decide which camera block and which landmark block each residual will touch.",
                code: [
                    "input = {",
                    "    \"camera_indices\": obs_to_camera,",
                    "    \"point_indices\": obs_to_point,",
                    "}",
                ],
            },
            {
                key: "replication",
                label: "Replication",
                caption: "The replication stage is just those indices being used to gather the active camera and landmark blocks.",
                code: [
                    "camera = camera_params[input[\"camera_indices\"]]",
                    "point = points_3d[input[\"point_indices\"]]",
                ],
            },
            {
                key: "residuals",
                label: "Residuals",
                caption: "Each gathered camera-landmark pair produces one reprojection residual.",
                code: [
                    "model = Reproj(camera_params, points_3d)",
                    "residual = model(points_2d, input[\"camera_indices\"], input[\"point_indices\"])",
                ],
            },
            {
                key: "jacobian",
                label: "Jacobian",
                caption: "Inside the optimizer step, that residual graph is differentiated into sparse Jacobian blocks.",
                code: [
                    "loss = optimizer.step(input)",
                    "# sparse Jacobian is assembled here",
                ],
            },
            {
                key: "complete",
                label: "Solve",
                caption: "After the Jacobian is built, the sparse solve runs and the pattern repeats on the next iteration.",
                code: [
                    "for _ in range(num_steps):",
                    "    loss = optimizer.step(input)",
                ],
            },
        ],
    },
    gauge: {
        intro: {
            label: "Ready",
            caption: "Gauge-fixed BA uses the same clickable step cards, but one camera is held fixed so its Jacobian column block disappears.",
        },
        phases: [
            {
                key: "inputs",
                label: "Fixed Gauge",
                caption: "The first camera is split out and held fixed before optimization starts.",
                code: [
                    "camera_fixed = camera_se3[:1].clone()",
                    "camera_trainable = camera_se3[1:]",
                ],
            },
            {
                key: "replication",
                label: "Replication",
                caption: "The fixed camera is concatenated back only for lookup; the trainable camera state is still just pose_rest.",
                code: [
                    "camera_all = torch.cat([camera_fixed, self.pose_rest], dim=0)",
                ],
            },
            {
                key: "residuals",
                label: "Residuals",
                caption: "Residuals still come from the selected point, selected camera pose, and selected intrinsics.",
                code: [
                    "residual = project_with_se3_and_intrinsics(",
                    "    self.points_3d[point_indices],",
                    "    camera_all[camera_indices],",
                    "    self.intrinsics[camera_indices],",
                    ") - points_2d",
                ],
            },
            {
                key: "jacobian",
                label: "Jacobian",
                caption: "The Jacobian is taken only with respect to the unfixed camera poses, intrinsics, and landmarks.",
                code: [
                    "J_cam_rest, J_intr, J_pts = autograd_graph.jacobian(",
                    "    residual, [model.pose_rest, model.intrinsics, model.points_3d]",
                    ")",
                ],
            },
            {
                key: "complete",
                label: "Gauge-Free",
                caption: "That is why the camera columns are reindexed to skip camera 0.",
                code: [
                    "expected_cam_cols = camera_idx[camera_idx > 0] - 1",
                ],
            },
        ],
    },
    pgo: {
        intro: {
            label: "Ready",
            caption: "The PGO tab keeps the same clickable structure so readers can hop between edge selection, residual construction, and the solve.",
        },
        phases: [
            {
                key: "inputs",
                label: "Inputs",
                caption: "The input is just edge connectivity, relative poses, and information weights.",
                code: [
                    "input = {\"edges\": edges, \"poses\": poses, \"infos\": infos}",
                ],
            },
            {
                key: "replication",
                label: "Replication",
                caption: "Each edge selects the two pose blocks it connects.",
                code: [
                    "node1 = self.nodes[edges[..., 0]]",
                    "node2 = self.nodes[edges[..., 1]]",
                ],
            },
            {
                key: "residuals",
                label: "Residuals",
                caption: "Those two selected poses and the measurement produce one weighted SE(3) residual.",
                code: [
                    "@map_transform",
                    "def _tracked_pose_graph_residual(poses, node1, node2, infos):",
                    "    residual = (pp.SE3(poses).Inv() @ pp.SE3(node1).Inv() @ pp.SE3(node2)).Log().tensor()",
                    "    return (infos @ residual[..., None])[..., 0]",
                ],
            },
            {
                key: "jacobian",
                label: "Jacobian",
                caption: "The optimizer differentiates that residual graph into sparse Jacobian blocks.",
                code: [
                    "residual = _tracked_pose_graph_residual(poses, node1, node2, infos)",
                    "loss = optimizer.step(input=input, weight=infos)",
                ],
            },
            {
                key: "complete",
                label: "Solve",
                caption: "Then the sparse solve runs and the same pattern repeats for the next iteration.",
                code: [
                    "for _ in range(num_steps):",
                    "    loss = optimizer.step(input=input, weight=infos)",
                ],
            },
        ],
    },
};

class CodePanel {
    constructor(sceneKey) {
        this.sceneKey = sceneKey;
        this.definition = CODE_SNIPPETS[sceneKey];
        this.block = document.getElementById(`${sceneKey}-code-block`);
        this.caption = document.getElementById(`${sceneKey}-code-caption`);
        this.stepPill = document.getElementById(`${sceneKey}-step-pill`);
        this.cardNodes = new Map();
        this.activePhase = "idle";
        this.phaseSelectHandler = null;
        this.render();
        this.setPhase("idle");
    }

    render() {
        let lineNumber = 1;
        const html = this.definition.phases.map((phase) => {
            const codeHtml = phase.code
                .map((line) => {
                    const renderedLine = renderPythonLine(line, lineNumber);
                    lineNumber += 1;
                    return renderedLine;
                })
                .join("");

            return [
                `<button class="phase-card" type="button" data-phase="${phase.key}">`,
                `<span class="phase-card-top">`,
                `<span class="phase-card-label">${phase.label}</span>`,
                `<span class="phase-card-jump">Jump</span>`,
                `</span>`,
                `<span class="phase-card-code">${codeHtml}</span>`,
                `</button>`,
            ].join("");
        }).join("");

        this.block.innerHTML = html;
        this.cardNodes = new Map(
            this.definition.phases.map((phase) => [
                phase.key,
                this.block.querySelector(`.phase-card[data-phase="${phase.key}"]`),
            ]),
        );

        this.cardNodes.forEach((node, phaseKey) => {
            node.addEventListener("click", () => {
                if (this.phaseSelectHandler) {
                    this.phaseSelectHandler(phaseKey);
                }
            });
        });
    }

    setPhaseSelectHandler(handler) {
        this.phaseSelectHandler = handler;
    }

    setPhase(phaseKey) {
        const phase = this.definition.phases.find((candidate) => candidate.key === phaseKey);
        const current = phase || this.definition.intro;
        const shouldDimInactiveCards = Boolean(phase && phase.key !== "complete");
        this.activePhase = phaseKey;

        this.stepPill.textContent = current.label;
        this.caption.textContent = current.caption;
        this.block.classList.toggle("has-active", shouldDimInactiveCards);

        let activeNode = null;
        this.cardNodes.forEach((node, key) => {
            const isActive = key === phaseKey;
            node.classList.toggle("active", isActive);
            if (isActive) {
                activeNode = node;
            }
        });

        if (activeNode) {
            activeNode.scrollIntoView({ block: "nearest" });
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
        this.codePanel.setPhaseSelectHandler((phaseKey) => this.seekToPhase(phaseKey));
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

    getPhaseOrder() {
        return [];
    }

    seekToPhase(phaseKey) {
        const phaseOrder = this.getPhaseOrder();
        const targetIndex = phaseOrder.indexOf(phaseKey);
        this.reset();
        if (targetIndex === -1) {
            return;
        }
        for (let index = 0; index <= targetIndex; index += 1) {
            this.applyPhase(phaseOrder[index]);
        }
        this.codePanel.setPhase(phaseKey);
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

    getPhaseOrder() {
        return ["inputs", "replication", "residuals", "jacobian", "complete"];
    }

    applyPhase(phaseKey) {
        switch (phaseKey) {
        case "inputs":
            this.elements.edges.forEach((edge) => {
                edge.style.opacity = "0.8";
                edge.style.transition = "stroke-dashoffset 1s ease-in-out";
                edge.style.strokeDashoffset = "0";
            });
            break;
        case "replication":
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
            break;
        case "residuals":
            this.elements.res.forEach((node) => {
                node.rect.style.opacity = "1";
                node.text.style.opacity = "1";
            });
            break;
        case "jacobian":
            [...this.elements.jacZEdges, ...this.elements.jacPEdges].forEach((edge) => {
                edge.style.opacity = "0.8";
                edge.style.transition = "stroke-dashoffset 1s ease-in-out";
                edge.style.strokeDashoffset = "0";
            });
            break;
        case "complete":
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
            [...this.elements.jacZEdges, ...this.elements.jacPEdges].forEach((edge) => {
                edge.style.opacity = "0";
            });
            break;
        default:
            break;
        }
    }

    play() {
        this.reset();
        let elapsed = 0;
        const timeline = [
            { key: "inputs", delay: 100 },
            { key: "replication", delay: 1200 },
            { key: "residuals", delay: 800 },
            { key: "jacobian", delay: 800 },
            { key: "complete", delay: 1200 },
        ];

        timeline.forEach((step) => {
            elapsed += step.delay;
            this.timeouts.push(setTimeout(() => {
                this.applyPhase(step.key);
                this.codePanel.setPhase(step.key);
            }, elapsed));
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

    getPhaseOrder() {
        return ["inputs", "replication", "residuals", "jacobian", "complete"];
    }

    applyPhase(phaseKey) {
        switch (phaseKey) {
        case "inputs":
            this.elements.edges.forEach((edge) => {
                edge.style.opacity = "0.8";
                edge.style.transition = "stroke-dashoffset 1s ease-in-out";
                edge.style.strokeDashoffset = "0";
            });
            break;
        case "replication":
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
            break;
        case "residuals":
            this.elements.res.forEach((node) => {
                node.rect.style.opacity = "1";
                node.text.style.opacity = "1";
            });
            break;
        case "jacobian":
            this.elements.jacEdges.forEach((edge) => {
                edge.style.opacity = "0.8";
                edge.style.transition = "stroke-dashoffset 1s ease-in-out";
                edge.style.strokeDashoffset = "0";
            });
            break;
        case "complete":
            this.elements.jacBlocks.forEach((entry) => {
                entry.block.style.opacity = entry.opacity;
                if (entry.mathNode) {
                    setMathNodeVisibility(entry.mathNode, true);
                }
            });
            this.elements.jacEdges.forEach((edge) => {
                edge.style.opacity = "0";
            });
            break;
        default:
            break;
        }
    }

    play() {
        this.reset();
        let elapsed = 0;
        const timeline = [
            { key: "inputs", delay: 100 },
            { key: "replication", delay: 1200 },
            { key: "residuals", delay: 800 },
            { key: "jacobian", delay: 800 },
            { key: "complete", delay: 1200 },
        ];

        timeline.forEach((step) => {
            elapsed += step.delay;
            this.timeouts.push(setTimeout(() => {
                this.applyPhase(step.key);
                this.codePanel.setPhase(step.key);
            }, elapsed));
        });
    }
}

class PerformanceChart {
    constructor(containerId, replayButtonId) {
        this.container = document.getElementById(containerId);
        this.replayButton = document.getElementById(replayButtonId);
        this.timeouts = [];
        this.hasPlayed = false;
        this.series = [
            {
                key: "g2o",
                label: "Speedup vs. G2O",
                trendLabel: "G2O Speedup Trend",
                color: "#8d6ad7",
                shape: "circle",
                points: [
                    [10500, 0.08], [11500, 0.7], [15000, 0.55], [38000, 5.2], [39000, 3.5], [52000, 3.3],
                    [90000, 13], [100000, 28], [105000, 20], [130000, 24], [250000, 10], [255000, 185], [335000, 125],
                ],
                trend: [
                    [10500, 0.16], [15000, 0.55], [36000, 4.8], [90000, 14], [110000, 23], [140000, 28], [250000, 72], [335000, 122],
                ],
            },
            {
                key: "gtsam",
                label: "Speedup vs. GTSAM",
                trendLabel: "GTSAM Speedup Trend",
                color: "#5f8ee7",
                shape: "square",
                points: [
                    [10500, 1.1], [11500, 7.8], [15000, 2.7], [35000, 4.7], [39000, 6.0], [85000, 18],
                    [100000, 6.5], [110000, 22], [130000, 7.0], [250000, 18], [255000, 95], [335000, 170],
                ],
                trend: [
                    [10500, 2.4], [15000, 3.8], [35000, 6.2], [90000, 9.2], [140000, 14], [250000, 50], [335000, 160],
                ],
            },
            {
                key: "ceres",
                label: "Speedup vs. Ceres",
                trendLabel: "Ceres Speedup Trend",
                color: "#0f8a8d",
                shape: "triangle",
                points: [
                    [10500, 0.7], [11500, 1.7], [15000, 1.7], [35000, 2.0], [35000, 5.0], [38000, 11.0],
                    [52000, 8.2], [90000, 13], [105000, 24], [110000, 28], [130000, 22], [250000, 19], [255000, 110], [335000, 140],
                ],
                trend: [
                    [10500, 0.95], [15000, 1.7], [35000, 5.0], [90000, 13.5], [140000, 26], [250000, 85], [335000, 135],
                ],
            },
        ];

        if (!this.container) {
            return;
        }

        this.render();
        this.replayButton?.addEventListener("click", () => this.play(true));

        if ("IntersectionObserver" in window) {
            this.observer = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting && !this.hasPlayed) {
                        this.play(false);
                    }
                });
            }, { threshold: 0.35 });
            this.observer.observe(this.container);
        } else {
            this.play(false);
        }
    }

    createSvgElement(tagName, attributes = {}) {
        const node = document.createElementNS(SVG_NS, tagName);
        Object.entries(attributes).forEach(([key, value]) => {
            node.setAttribute(key, value);
        });
        return node;
    }

    xScale(value) {
        const domainMin = Math.log10(10000);
        const domainMax = Math.log10(350000);
        const rangeMin = 110;
        const rangeMax = 900;
        const normalized = (Math.log10(value) - domainMin) / (domainMax - domainMin);
        return rangeMin + normalized * (rangeMax - rangeMin);
    }

    yScale(value) {
        const domainMin = Math.log10(0.05);
        const domainMax = Math.log10(300);
        const rangeMin = 470;
        const rangeMax = 45;
        const normalized = (Math.log10(value) - domainMin) / (domainMax - domainMin);
        return rangeMin - normalized * (rangeMin - rangeMax);
    }

    buildSmoothPath(points) {
        const mapped = points.map(([x, y]) => [this.xScale(x), this.yScale(y)]);
        if (mapped.length === 0) {
            return "";
        }
        if (mapped.length === 1) {
            return `M ${mapped[0][0]} ${mapped[0][1]}`;
        }

        let path = `M ${mapped[0][0]} ${mapped[0][1]}`;
        for (let index = 0; index < mapped.length - 1; index += 1) {
            const current = mapped[index];
            const next = mapped[index + 1];
            const previous = mapped[index - 1] || current;
            const afterNext = mapped[index + 2] || next;
            const cp1x = current[0] + (next[0] - previous[0]) / 6;
            const cp1y = current[1] + (next[1] - previous[1]) / 6;
            const cp2x = next[0] - (afterNext[0] - current[0]) / 6;
            const cp2y = next[1] - (afterNext[1] - current[1]) / 6;
            path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${next[0]} ${next[1]}`;
        }
        return path;
    }

    buildPointShape(shape, x, y, color) {
        if (shape === "square") {
            return this.createSvgElement("rect", {
                x: x - 7,
                y: y - 7,
                width: 14,
                height: 14,
                fill: color,
                class: "perf-point",
            });
        }

        if (shape === "triangle") {
            return this.createSvgElement("path", {
                d: `M ${x} ${y - 8} L ${x - 8} ${y + 7} L ${x + 8} ${y + 7} Z`,
                fill: color,
                class: "perf-point",
            });
        }

        return this.createSvgElement("circle", {
            cx: x,
            cy: y,
            r: 7,
            fill: color,
            class: "perf-point",
        });
    }

    render() {
        this.container.innerHTML = "";
        this.svg = this.createSvgElement("svg", {
            viewBox: "0 0 960 560",
            preserveAspectRatio: "xMidYMid meet",
            "aria-hidden": "true",
        });

        const staticGroup = this.createSvgElement("g");
        const trendGroup = this.createSvgElement("g");
        const pointsGroup = this.createSvgElement("g");

        this.staticNodes = [];
        this.trendPaths = [];
        this.dotNodes = [];

        const yTicks = [
            { value: 0.1, label: "10^-1" },
            { value: 1, label: "10^0" },
            { value: 10, label: "10^1" },
            { value: 100, label: "10^2" },
        ];
        const xTicks = [
            { value: 10000, label: "10k" },
            { value: 30000, label: "30k" },
            { value: 100000, label: "100k" },
            { value: 300000, label: "300k" },
        ];

        yTicks.forEach((tick) => {
            const y = this.yScale(tick.value);
            const line = this.createSvgElement("line", {
                x1: 110,
                y1: y,
                x2: 900,
                y2: y,
                class: "perf-grid-line",
            });
            const label = this.createSvgElement("text", {
                x: 92,
                y: y + 5,
                "text-anchor": "end",
                class: "perf-tick-label",
            });
            label.textContent = tick.label;
            this.staticNodes.push(line, label);
            staticGroup.append(line, label);
        });

        xTicks.forEach((tick) => {
            const x = this.xScale(tick.value);
            const line = this.createSvgElement("line", {
                x1: x,
                y1: 45,
                x2: x,
                y2: 470,
                class: "perf-grid-line",
            });
            const label = this.createSvgElement("text", {
                x,
                y: 500,
                "text-anchor": "middle",
                class: "perf-tick-label",
            });
            label.textContent = tick.label;
            this.staticNodes.push(line, label);
            staticGroup.append(line, label);
        });

        const yAxis = this.createSvgElement("line", {
            x1: 110, y1: 45, x2: 110, y2: 470, class: "perf-axis-line",
        });
        const xAxis = this.createSvgElement("line", {
            x1: 110, y1: 470, x2: 900, y2: 470, class: "perf-axis-line",
        });
        const xLabel = this.createSvgElement("text", {
            x: 505, y: 540, "text-anchor": "middle", class: "perf-axis-label",
        });
        xLabel.textContent = "Number of Optimizable Parameters";
        const yLabel = this.createSvgElement("text", {
            x: 28, y: 260, "text-anchor": "middle", class: "perf-axis-label",
            transform: "rotate(-90 28 260)",
        });
        yLabel.textContent = "Speedup";
        this.staticNodes.push(yAxis, xAxis, xLabel, yLabel);
        staticGroup.append(yAxis, xAxis, xLabel, yLabel);

        this.series.forEach((series, seriesIndex) => {
            const trendPath = this.createSvgElement("path", {
                d: this.buildSmoothPath(series.trend),
                class: "perf-trend",
                stroke: series.color,
            });
            trendGroup.appendChild(trendPath);
            this.trendPaths.push(trendPath);

            series.points.forEach(([xValue, yValue], pointIndex) => {
                const x = this.xScale(xValue);
                const y = this.yScale(yValue);
                const group = this.createSvgElement("g", { class: "perf-dot" });
                group.style.transformOrigin = `${x}px ${y}px`;
                group.appendChild(this.buildPointShape(series.shape, x, y, series.color));
                pointsGroup.appendChild(group);
                this.dotNodes.push({ node: group, delay: 420 + (seriesIndex * 120) + (pointIndex * 45) });
            });
        });

        const legendX = 520;
        const legendY = 258;
        const legend = this.createSvgElement("rect", {
            x: legendX,
            y: legendY,
            width: 332,
            height: 208,
            rx: 12,
            class: "perf-legend",
        });
        this.staticNodes.push(legend);
        staticGroup.appendChild(legend);

        this.series.forEach((series, index) => {
            const y = legendY + 30 + index * 58;
            const label = this.createSvgElement("text", {
                x: legendX + 74,
                y: y + 5,
                class: "perf-legend-label",
            });
            label.textContent = series.label;
            const marker = this.buildPointShape(series.shape, legendX + 34, y, series.color);
            const trendLine = this.createSvgElement("line", {
                x1: legendX + 12,
                y1: y + 24,
                x2: legendX + 54,
                y2: y + 24,
                stroke: series.color,
                "stroke-width": 4,
                "stroke-linecap": "round",
            });
            const trendLabel = this.createSvgElement("text", {
                x: legendX + 74,
                y: y + 29,
                class: "perf-legend-label",
            });
            trendLabel.textContent = series.trendLabel;
            this.staticNodes.push(marker, label, trendLine, trendLabel);
            staticGroup.append(marker, label, trendLine, trendLabel);
        });

        this.svg.append(staticGroup, trendGroup, pointsGroup);
        this.container.appendChild(this.svg);
        this.resetAnimationState();
    }

    clearTimers() {
        this.timeouts.forEach((timeoutId) => clearTimeout(timeoutId));
        this.timeouts = [];
    }

    resetAnimationState() {
        this.clearTimers();
        this.staticNodes.forEach((node) => {
            node.style.transition = "none";
            node.style.opacity = "0";
        });
        this.trendPaths.forEach((path) => {
            const length = path.getTotalLength();
            path.style.transition = "none";
            path.style.strokeDasharray = `${length}`;
            path.style.strokeDashoffset = `${length}`;
            path.style.opacity = "1";
        });
        this.dotNodes.forEach(({ node }) => {
            node.style.transition = "none";
            node.style.opacity = "0";
            node.style.transform = "translateY(14px) scale(0.82)";
        });
    }

    play(forceReplay) {
        if (!forceReplay && this.hasPlayed) {
            return;
        }

        this.hasPlayed = true;
        this.resetAnimationState();

        window.requestAnimationFrame(() => {
            this.staticNodes.forEach((node) => {
                node.style.transition = "opacity 420ms ease";
                node.style.opacity = "1";
            });

            this.trendPaths.forEach((path, index) => {
                const startDelay = 220 + index * 220;
                this.timeouts.push(setTimeout(() => {
                    path.style.transition = "stroke-dashoffset 1000ms cubic-bezier(0.23, 1, 0.32, 1)";
                    path.style.strokeDashoffset = "0";
                }, startDelay));
            });

            this.dotNodes.forEach(({ node, delay }) => {
                this.timeouts.push(setTimeout(() => {
                    node.style.transition = "opacity 260ms ease, transform 360ms ease";
                    node.style.opacity = "1";
                    node.style.transform = "translateY(0) scale(1)";
                }, delay));
            });
        });
    }
}

class VideoLoopCarousel {
    constructor(rootId) {
        this.root = document.getElementById(rootId);
        if (!this.root) {
            return;
        }

        this.track = this.root.querySelector(".video-loop-track");
        this.slides = Array.from(this.root.querySelectorAll(".video-slide"));
        this.navButtons = Array.from(this.root.querySelectorAll(".video-loop-dot"));
        this.activeIndex = 0;

        this.navButtons.forEach((button) => {
            button.addEventListener("click", () => {
                this.showSlide(Number(button.dataset.slideIndex || 0), true);
            });
        });

        this.slides.forEach((slide, index) => {
            const video = slide.querySelector("video");
            if (!video) {
                return;
            }
            video.addEventListener("ended", () => {
                if (index === this.activeIndex) {
                    this.showSlide(index + 1, true);
                }
            });
        });

        this.showSlide(0, false);
    }

    showSlide(index, restartVideo) {
        if (!this.root || this.slides.length === 0) {
            return;
        }

        this.activeIndex = ((index % this.slides.length) + this.slides.length) % this.slides.length;
        this.track.style.transform = `translate3d(${-this.activeIndex * 100}%, 0, 0)`;

        this.slides.forEach((slide, slideIndex) => {
            const isActive = slideIndex === this.activeIndex;
            slide.classList.toggle("active", isActive);
            const video = slide.querySelector("video");
            if (!video) {
                return;
            }
            if (isActive) {
                if (restartVideo) {
                    video.currentTime = 0;
                }
                const playPromise = video.play();
                if (playPromise && typeof playPromise.catch === "function") {
                    playPromise.catch(() => {});
                }
            } else {
                video.pause();
                video.currentTime = 0;
            }
        });
        this.navButtons.forEach((button, buttonIndex) => {
            button.classList.toggle("active", buttonIndex === this.activeIndex);
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

    new PerformanceChart("performance-chart", "performance-replay");
    new VideoLoopCarousel("product-story");
});
