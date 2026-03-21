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
            { number: 41, text: "input = {" },
            { number: 42, text: "    \"points_2d\": trimmed_dataset['points_2d']," },
            { number: 43, text: "    \"camera_indices\": trimmed_dataset['camera_index_of_observations']," },
            { number: 44, text: "    \"point_indices\": trimmed_dataset['point_index_of_observations']" },
            { number: 45, text: "}" },
            { number: 46, text: "" },
            { number: 47, text: "model = Reproj(" },
            { number: 48, text: "    trimmed_dataset['camera_params'][:, :NUM_CAMERA_PARAMS].clone()," },
            { number: 49, text: "    trimmed_dataset['points_3d'].clone()" },
            { number: 50, text: ").to(DEVICE)" },
            { number: 51, text: "strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)" },
            { number: 52, text: "solver = PCG(tol=1e-4, maxiter=250)  # or CuDSS()" },
            { number: 53, text: "optimizer = LM(model, strategy=strategy, solver=solver, reject=30)" },
            { number: 54, text: "" },
            { number: 65, text: "start = perf_counter()" },
            { number: 66, text: "for idx in range(20):" },
            { number: 67, text: "    loss = optimizer.step(input)" },
            { number: 68, text: "    print('Iteration', idx, 'loss', loss.item(), 'time', perf_counter() - start)" },
        ],
        phases: {
            idle: {
                label: "Ready",
                caption: "Play the BA animation to walk from observation indices to the sparse Jacobian assembled inside each optimizer step.",
                lines: [],
            },
            inputs: {
                label: "Inputs",
                caption: "Each observation contributes a 2D target plus the camera and landmark indices that decide which state blocks are touched.",
                lines: [41, 42, 43, 44, 45],
            },
            replication: {
                label: "Replication",
                caption: "Those camera and point index arrays are exactly what the replication stage in the animation is visualizing.",
                lines: [43, 44],
            },
            residuals: {
                label: "Residuals",
                caption: "The `Reproj` module pairs the selected camera block with the selected landmark block to form each reprojection residual.",
                lines: [47, 48, 49, 50],
            },
            jacobian: {
                label: "Jacobian",
                caption: "Calling `LM(...)` and then `optimizer.step(input)` triggers the sparsity-aware autograd path that populates the Jacobian blocks.",
                lines: [51, 52, 53, 67],
            },
            complete: {
                label: "Solve",
                caption: "After the sparse Jacobian is built, the optimizer uses it inside the trust-region step and repeats for the next iteration.",
                lines: [53, 65, 66, 67, 68],
            },
        },
    },
    gauge: {
        lines: [
            { number: 353, text: "        self.pose_rest = nn.Parameter(TrackingTensor(camera_se3_rest))" },
            { number: 354, text: "        self.intrinsics = nn.Parameter(TrackingTensor(camera_intrinsics))" },
            { number: 355, text: "        self.points_3d = nn.Parameter(TrackingTensor(points_3d))" },
            { number: 356, text: "        self.pose_rest.trim_SE3_grad = True" },
            { number: 357, text: "" },
            { number: 358, text: "    def forward(self, points_2d, camera_indices, point_indices, camera_fixed):" },
            { number: 359, text: "        camera_se3 = torch.cat([camera_fixed, self.pose_rest], dim=0)" },
            { number: 360, text: "        points_proj = project_with_se3_and_intrinsics(" },
            { number: 361, text: "            self.points_3d[point_indices]," },
            { number: 362, text: "            camera_se3[camera_indices]," },
            { number: 363, text: "            self.intrinsics[camera_indices]," },
            { number: 364, text: "        )" },
            { number: 365, text: "        return points_proj - points_2d" },
            { number: 366, text: "" },
            { number: 393, text: "    camera_fixed = camera_se3[:1].clone()" },
            { number: 394, text: "    input = {" },
            { number: 395, text: "        \"points_2d\": points_2d," },
            { number: 396, text: "        \"camera_indices\": camera_idx," },
            { number: 397, text: "        \"point_indices\": point_idx," },
            { number: 398, text: "        \"camera_fixed\": camera_fixed," },
            { number: 399, text: "    }" },
            { number: 400, text: "" },
            { number: 401, text: "    model = ReprojFixedFirstCameraCat(" },
            { number: 402, text: "        camera_se3[1:].clone()," },
            { number: 403, text: "        camera_intrinsics.clone()," },
            { number: 404, text: "        points_3d.clone()," },
            { number: 405, text: "    )" },
            { number: 406, text: "    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)" },
            { number: 407, text: "    solver = PCG(tol=1e-4, maxiter=250)" },
            { number: 408, text: "    optimizer = LM(model, strategy=strategy, solver=solver, reject=30)" },
            { number: 409, text: "" },
            { number: 410, text: "    for _ in range(20):" },
            { number: 411, text: "        optimizer.step(input)" },
            { number: 412, text: "" },
            { number: 525, text: "    J_cam_rest, J_intr, J_pts = autograd_graph.jacobian(" },
            { number: 526, text: "        residual," },
            { number: 527, text: "        [model.pose_rest, model.intrinsics, model.points_3d]," },
            { number: 528, text: "    )" },
            { number: 529, text: "    assert J_cam_rest.layout == torch.sparse_bsr" },
            { number: 530, text: "    assert J_intr.layout == torch.sparse_bsr" },
            { number: 531, text: "    assert J_pts.layout == torch.sparse_bsr" },
            { number: 532, text: "" },
            { number: 541, text: "    expected_cam_cols = (camera_idx[camera_idx > 0] - 1).to(dtype=J_cam_rest.col_indices().dtype)" },
            { number: 542, text: "    assert torch.equal(J_cam_rest.col_indices(), expected_cam_cols)" },
            { number: 543, text: "    assert torch.unique(J_cam_rest.col_indices()).numel() == n_cams_rest" },
            { number: 544, text: "    assert torch.equal(J_intr.col_indices(), camera_idx)" },
            { number: 545, text: "    assert torch.unique(J_intr.col_indices()).numel() == n_cams_intr" },
            { number: 546, text: "    assert torch.equal(J_pts.col_indices(), point_idx)" },
        ],
        phases: {
            idle: {
                label: "Ready",
                caption: "Play the gauge-fixed BA animation to see how the fixed first camera is excluded from one Jacobian branch while intrinsics and points remain active.",
                lines: [],
            },
            inputs: {
                label: "Fixed Gauge",
                caption: "The gauge-fixed variant slices out `camera_fixed` and passes it alongside the usual observation-index tensors.",
                lines: [393, 394, 395, 396, 397, 398, 399, 401, 402, 403, 404, 405],
            },
            replication: {
                label: "Replication",
                caption: "Inside `forward`, the fixed camera is concatenated back only for lookup, but the optimizable camera state is still just `pose_rest`.",
                lines: [358, 359, 360, 361, 362, 363, 364],
            },
            residuals: {
                label: "Residuals",
                caption: "Residuals are still computed from the selected point, selected camera SE(3), and selected intrinsics, exactly like the replicated nodes suggest.",
                lines: [360, 361, 362, 363, 364, 365],
            },
            jacobian: {
                label: "Jacobian",
                caption: "The sparse Jacobian is requested only for `pose_rest`, `intrinsics`, and `points_3d`, so the fixed first camera never gets a Jacobian block.",
                lines: [525, 526, 527, 528, 529, 530, 531],
            },
            complete: {
                label: "Gauge-Free",
                caption: "These assertions verify the structural effect of gauge fixing: camera columns are reindexed to skip camera 0, while intrinsics and landmark columns stay aligned with all observations.",
                lines: [541, 542, 543, 544, 545, 546],
            },
        },
    },
    pgo: {
        lines: [
            { number: 99, text: "@map_transform" },
            { number: 100, text: "def foo(poses, node1, node2, infos):" },
            { number: 101, text: "    residual = (pp.SE3(poses).Inv() @ pp.SE3(node1).Inv() @ pp.SE3(node2)).Log().tensor()" },
            { number: 102, text: "    residual = infos @ residual[..., None]" },
            { number: 103, text: "    residual = residual[..., 0]" },
            { number: 104, text: "    return residual" },
            { number: 105, text: "" },
            { number: 106, text: "class PoseGraph(nn.Module):" },
            { number: 107, text: "" },
            { number: 108, text: "    def __init__(self, nodes):" },
            { number: 109, text: "        super().__init__()" },
            { number: 110, text: "        self.nodes = nn.Parameter(TrackingTensor(nodes))" },
            { number: 111, text: "        self.nodes.trim_SE3_grad = True" },
            { number: 112, text: "" },
            { number: 113, text: "    def forward(self, edges, poses, infos):" },
            { number: 114, text: "        node1 = self.nodes[edges[..., 0]]" },
            { number: 115, text: "        node2 = self.nodes[edges[..., 1]]" },
            { number: 116, text: "        return foo(poses, node1, node2, infos)" },
            { number: 117, text: "" },
            { number: 145, text: "    edges, poses, infos = data.edges, data.poses, data.infos" },
            { number: 146, text: "    infos = torch.linalg.cholesky(infos)" },
            { number: 147, text: "    input = {'edges': edges, 'poses': poses, 'infos': infos}" },
            { number: 148, text: "" },
            { number: 149, text: "    graph = PoseGraph(data.nodes).to(args.device)" },
            { number: 150, text: "    # solver = PCG(tol=1e-5)" },
            { number: 151, text: "    solver = cuSolverSP()" },
            { number: 152, text: "    # solver = ppos.Cholesky()" },
            { number: 153, text: "    # strategy = ppost.TrustRegion(radius=1e4, min=1e-32, max=1e16)" },
            { number: 154, text: "    # strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)" },
            { number: 155, text: "    strategy = pp.optim.strategy.Adaptive()" },
            { number: 156, text: "" },
            { number: 157, text: "    optimizer = LM(graph, solver=solver, strategy=strategy, min=1e-10, reject=30)" },
            { number: 158, text: "    scheduler = StopOnPlateau(optimizer, steps=20, patience=3, decreasing=1e-7, verbose=True)" },
            { number: 159, text: "" },
            { number: 167, text: "    for i in range(10):" },
            { number: 168, text: "        loss = optimizer.step(input=input, weight=infos)" },
        ],
        phases: {
            idle: {
                label: "Ready",
                caption: "Play the PGO animation to trace each edge from indexed pose selection through residual assembly and into the sparse solve.",
                lines: [],
            },
            inputs: {
                label: "Inputs",
                caption: "The pose-graph step starts from edge indices, measured relative poses, and information matrices packed into the optimizer input.",
                lines: [145, 146, 147, 149],
            },
            replication: {
                label: "Replication",
                caption: "Each edge chooses two pose blocks by indexing into `self.nodes`, which is exactly what the duplicated pose nodes in the animation represent.",
                lines: [110, 111, 113, 114, 115, 116],
            },
            residuals: {
                label: "Residuals",
                caption: "The mapped `foo(...)` function computes one SE(3) residual per edge and weights it with the information matrix.",
                lines: [99, 100, 101, 102, 103, 104],
            },
            jacobian: {
                label: "Jacobian",
                caption: "Once the residual graph is defined, `LM(...)` and `optimizer.step(...)` differentiate through it to form the sparse Jacobian blocks.",
                lines: [149, 151, 155, 157, 167, 168],
            },
            complete: {
                label: "Solve",
                caption: "The highlighted optimizer loop is where the Jacobian is assembled, solved, and then reused on the next pose-graph iteration.",
                lines: [157, 158, 167, 168],
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
