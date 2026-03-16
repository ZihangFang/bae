const SVG_NS = "http://www.w3.org/2000/svg";

document.addEventListener('DOMContentLoaded', () => {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Initialize scenes
    const baScene = new BACanvas('ba-viz', false);
    const gaugeScene = new BACanvas('gauge-viz', true);
    const pgoScene = new PGOCanvas('pgo-viz');
    
    document.getElementById('play-ba').addEventListener('click', () => baScene.play());
    document.getElementById('reset-ba').addEventListener('click', () => baScene.reset());

    document.getElementById('play-gauge').addEventListener('click', () => gaugeScene.play());
    document.getElementById('reset-gauge').addEventListener('click', () => gaugeScene.reset());

    document.getElementById('play-pgo').addEventListener('click', () => pgoScene.play());
    document.getElementById('reset-pgo').addEventListener('click', () => pgoScene.reset());
});

class BACanvas {
    constructor(containerId, isGaugeFixed) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.isGaugeFixed = isGaugeFixed;
        this.svg = document.createElementNS(SVG_NS, "svg");
        this.svg.setAttribute("viewBox", "0 0 1000 600");
        this.container.appendChild(this.svg);
        this.elements = {};
        this.timeline = null;
        this.build();
    }

    build() {
        this.svg.innerHTML = '';
        this.elements = {
            edges: [],
            repZ: [],
            repP: [],
            res: [],
            jacZEdges: [],
            jacPEdges: [],
            jacZBlocks: [],
            jacPBlocks: []
        };

        const g = document.createElementNS(SVG_NS, "g");
        this.svg.appendChild(g);
        this.mainGroup = g;

        const numZ = 2;
        const numP = 3;
        const obs = [
            { c: 0, p: 0 },
            { c: 0, p: 1 },
            { c: 1, p: 1 },
            { c: 1, p: 2 }
        ];

        const zX = 100, zStartY = 150, zGap = 60;
        const pX = 100, pStartY = 350, pGap = 60;
        const repZX = 350, repPX = 430, resX = 550, repStartY = 200, resGap = 60;
        const jzX = 750, jpX = 850, jStartY = 200;
        const blockSize = 30, blockGap = 4;

        this.addText("Z (Cameras)", zX + 20, zStartY - 40, "title-label");
        this.addText("P (Landmarks)", pX + 20, pStartY - 40, "title-label");
        this.addText("Replication", (repZX + repPX) / 2 + 20, repStartY - 40, "title-label");
        this.addText("Residuals R", resX + 20, repStartY - 40, "title-label");
        this.addText("∂R/∂Z", jzX + (numZ*blockSize)/2, jStartY - 40, "title-label");
        this.addText("∂R/∂P", jpX + (numP*blockSize)/2, jStartY - 40, "title-label");

        const zNodes = [];
        for (let i = 0; i < numZ; i++) {
            const isFixed = this.isGaugeFixed && i === 0;
            const node = this.addRect(zX, zStartY + i * zGap, 40, 40, `cam-node ${isFixed ? 'fixed-node' : ''}`);
            this.addText(`C${i}`, zX + 20, zStartY + i * zGap + 20, "label");
            zNodes.push({x: zX + 40, y: zStartY + i * zGap + 20, el: node});
        }
        const pNodes = [];
        for (let i = 0; i < numP; i++) {
            const node = this.addRect(pX, pStartY + i * pGap, 40, 40, "pt-node");
            this.addText(`P${i}`, pX + 20, pStartY + i * pGap + 20, "label");
            pNodes.push({x: pX + 40, y: pStartY + i * pGap + 20, el: node});
        }

        obs.forEach((o, i) => {
            const y = repStartY + i * resGap;
            const edgeZFlow = this.addPath(zNodes[o.c].x, zNodes[o.c].y, repZX, y + 20, "edge-flow");
            const edgePFlow = this.addPath(pNodes[o.p].x, pNodes[o.p].y, repPX, y + 20, "edge-flow");
            this.elements.edges.push(edgeZFlow, edgePFlow);

            const isFixed = this.isGaugeFixed && o.c === 0;
            const repZNode = this.addRect(repZX, y, 40, 40, `cam-node ${isFixed ? 'fixed-node' : ''}`);
            repZNode.style.opacity = 0;
            const repZText = this.addText(`C${o.c}`, repZX + 20, y + 20, "label");
            repZText.style.opacity = 0;
            this.elements.repZ.push({ rect: repZNode, text: repZText });

            const repPNode = this.addRect(repPX, y, 40, 40, "pt-node");
            repPNode.style.opacity = 0;
            const repPText = this.addText(`P${o.p}`, repPX + 20, y + 20, "label");
            repPText.style.opacity = 0;
            this.elements.repP.push({ rect: repPNode, text: repPText });

            this.addPath(repZX + 40, y + 20, resX, y + 20, "edge");
            this.addPath(repPX + 40, y + 20, resX, y + 20, "edge");

            const resNode = this.addRect(resX, y, 40, 40, "res-node");
            resNode.style.opacity = 0;
            const resText = this.addText(`r${i}`, resX + 20, y + 20, "label");
            resText.style.opacity = 0;
            this.elements.res.push({ rect: resNode, text: resText, y: y+20 });
        });

        for (let r = 0; r < obs.length; r++) {
            for (let c = 0; c < numZ; c++) {
                const isNonZero = obs[r].c === c;
                const isFixed = this.isGaugeFixed && c === 0;
                const opacity = (isNonZero && !isFixed) ? 1 : 0.1;
                const bx = jzX + c * (blockSize + blockGap);
                const by = jStartY + r * (blockSize + blockGap);
                const block = this.addRect(bx, by, blockSize, blockSize, "cam-jac-node node-rect");
                block.style.opacity = 0;
                this.elements.jacZBlocks.push({ block, opacity });
                if (isNonZero && !isFixed) {
                    const edge = this.addPath(resX + 40, this.elements.res[r].y, bx, by + blockSize/2, "edge-flow");
                    this.elements.jacZEdges.push(edge);
                }
            }
        }

        for (let r = 0; r < obs.length; r++) {
            for (let c = 0; c < numP; c++) {
                const isNonZero = obs[r].p === c;
                const opacity = isNonZero ? 1 : 0.1;
                const bx = jpX + c * (blockSize + blockGap);
                const by = jStartY + r * (blockSize + blockGap);
                const block = this.addRect(bx, by, blockSize, blockSize, "pt-jac-node node-rect");
                block.style.opacity = 0;
                this.elements.jacPBlocks.push({ block, opacity });
                if (isNonZero) {
                    const edge = this.addPath(resX + 40, this.elements.res[r].y, bx, by + blockSize/2, "edge-flow");
                    this.elements.jacPEdges.push(edge);
                }
            }
        }
    }

    addRect(x, y, w, h, classes) {
        const r = document.createElementNS(SVG_NS, "rect");
        r.setAttribute("x", x);
        r.setAttribute("y", y);
        r.setAttribute("width", w);
        r.setAttribute("height", h);
        r.setAttribute("class", `node-rect ${classes}`);
        this.mainGroup.appendChild(r);
        return r;
    }

    addText(content, x, y, classes) {
        const t = document.createElementNS(SVG_NS, "text");
        t.setAttribute("x", x);
        t.setAttribute("y", y);
        t.setAttribute("class", classes);
        t.textContent = content;
        this.mainGroup.appendChild(t);
        return t;
    }

    addPath(x1, y1, x2, y2, classes) {
        const p = document.createElementNS(SVG_NS, "path");
        const dx = Math.abs(x2 - x1) * 0.5;
        const d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
        p.setAttribute("d", d);
        p.setAttribute("class", classes);
        this.mainGroup.appendChild(p);
        
        if (classes.includes("edge-flow")) {
            const length = p.getTotalLength();
            p.style.strokeDasharray = length;
            p.style.strokeDashoffset = length;
        }
        return p;
    }

    reset() {
        if (this.timeline) clearTimeout(this.timeline);
        this.build();
    }

    play() {
        this.reset();
        let t = 0;
        const step = (fn, delay) => { t += delay; this.timeline = setTimeout(fn, t); };

        step(() => {
            this.elements.edges.forEach(e => {
                e.style.opacity = 0.8;
                e.style.transition = "stroke-dashoffset 1s ease-in-out";
                e.style.strokeDashoffset = "0";
            });
        }, 100);

        step(() => {
            this.elements.repZ.forEach(n => {
                n.rect.style.opacity = n.rect.classList.contains('fixed-node') ? 0.4 : 1;
                n.text.style.opacity = 1;
            });
            this.elements.repP.forEach(n => {
                n.rect.style.opacity = 1;
                n.text.style.opacity = 1;
            });
            this.elements.edges.forEach(e => e.style.opacity = 0);
        }, 1200);

        step(() => {
            this.elements.res.forEach(n => {
                n.rect.style.opacity = 1;
                n.text.style.opacity = 1;
            });
        }, 800);

        step(() => {
            [...this.elements.jacZEdges, ...this.elements.jacPEdges].forEach(e => {
                e.style.opacity = 0.8;
                e.style.transition = "stroke-dashoffset 1s ease-in-out";
                e.style.strokeDashoffset = "0";
            });
        }, 800);

        step(() => {
            this.elements.jacZBlocks.forEach(b => b.block.style.opacity = b.opacity);
            this.elements.jacPBlocks.forEach(b => b.block.style.opacity = b.opacity);
        }, 1200);

        step(() => {
            [...this.elements.jacZEdges, ...this.elements.jacPEdges].forEach(e => e.style.opacity = 0);
        }, 800);
    }
}

class PGOCanvas {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.svg = document.createElementNS(SVG_NS, "svg");
        this.svg.setAttribute("viewBox", "0 0 1000 600");
        this.container.appendChild(this.svg);
        this.elements = {};
        this.timeline = null;
        this.build();
    }

    build() {
        this.svg.innerHTML = '';
        this.elements = {
            edges: [],
            repNodes: [],
            res: [],
            jacEdges: [],
            jacBlocks: []
        };

        const g = document.createElementNS(SVG_NS, "g");
        this.svg.appendChild(g);
        this.mainGroup = g;

        const numNodes = 3;
        const edges = [
            { from: 0, to: 1 },
            { from: 1, to: 2 },
            { from: 0, to: 2 }
        ];

        const nX = 150, nStartY = 200, nGap = 80;
        const repX1 = 350, repX2 = 450, resX = 580, rStartY = 200, rGap = 80;
        const jX = 800, jStartY = 200;
        const blockSize = 30, blockGap = 4;

        this.addText("Poses X", nX + 20, nStartY - 40, "title-label");
        this.addText("Replication", (repX1 + repX2) / 2 + 20, rStartY - 40, "title-label");
        this.addText("Residuals R", resX + 20, rStartY - 40, "title-label");
        this.addText("∂R/∂X", jX + (numNodes*blockSize)/2, jStartY - 40, "title-label");

        const nodes = [];
        for (let i = 0; i < numNodes; i++) {
            const node = this.addRect(nX, nStartY + i * nGap, 40, 40, "cam-node");
            this.addText(`X${i}`, nX + 20, nStartY + i * nGap + 20, "label");
            nodes.push({x: nX + 40, y: nStartY + i * nGap + 20, el: node});
        }

        edges.forEach((o, i) => {
            const y = rStartY + i * rGap;
            const edgeFlow1 = this.addPath(nodes[o.from].x, nodes[o.from].y, repX1, y + 20, "edge-flow");
            const edgeFlow2 = this.addPath(nodes[o.to].x, nodes[o.to].y, repX2, y + 20, "edge-flow");
            this.elements.edges.push(edgeFlow1, edgeFlow2);

            const rep1Node = this.addRect(repX1, y, 40, 40, "cam-node");
            rep1Node.style.opacity = 0;
            const rep1Text = this.addText(`X${o.from}`, repX1 + 20, y + 20, "label");
            rep1Text.style.opacity = 0;
            
            const rep2Node = this.addRect(repX2, y, 40, 40, "cam-node");
            rep2Node.style.opacity = 0;
            const rep2Text = this.addText(`X${o.to}`, repX2 + 20, y + 20, "label");
            rep2Text.style.opacity = 0;
            
            this.elements.repNodes.push({ rect: rep1Node, text: rep1Text }, { rect: rep2Node, text: rep2Text });

            this.addPath(repX1 + 40, y + 20, resX, y + 20, "edge");
            this.addPath(repX2 + 40, y + 20, resX, y + 20, "edge");

            const resNode = this.addRect(resX, y, 40, 40, "res-node");
            resNode.style.opacity = 0;
            const resText = this.addText(`r${i}`, resX + 20, y + 20, "label");
            resText.style.opacity = 0;
            this.elements.res.push({ rect: resNode, text: resText, y: y+20 });
        });

        for (let r = 0; r < edges.length; r++) {
            for (let c = 0; c < numNodes; c++) {
                const isNonZero = (edges[r].from === c) || (edges[r].to === c);
                const opacity = isNonZero ? 1 : 0.1;
                const bx = jX + c * (blockSize + blockGap);
                const by = jStartY + r * (blockSize + blockGap);
                const block = this.addRect(bx, by, blockSize, blockSize, "cam-jac-node node-rect");
                block.style.opacity = 0;
                this.elements.jacBlocks.push({ block, opacity });
                if (isNonZero) {
                    const edge = this.addPath(resX + 40, this.elements.res[r].y, bx, by + blockSize/2, "edge-flow");
                    this.elements.jacEdges.push(edge);
                }
            }
        }
    }

    addRect(x, y, w, h, classes) {
        const r = document.createElementNS(SVG_NS, "rect");
        r.setAttribute("x", x);
        r.setAttribute("y", y);
        r.setAttribute("width", w);
        r.setAttribute("height", h);
        r.setAttribute("class", `node-rect ${classes}`);
        this.mainGroup.appendChild(r);
        return r;
    }

    addText(content, x, y, classes) {
        const t = document.createElementNS(SVG_NS, "text");
        t.setAttribute("x", x);
        t.setAttribute("y", y);
        t.setAttribute("class", classes);
        t.textContent = content;
        this.mainGroup.appendChild(t);
        return t;
    }

    addPath(x1, y1, x2, y2, classes) {
        const p = document.createElementNS(SVG_NS, "path");
        const dx = Math.abs(x2 - x1) * 0.5;
        const d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
        p.setAttribute("d", d);
        p.setAttribute("class", classes);
        this.mainGroup.appendChild(p);
        
        if (classes.includes("edge-flow")) {
            const length = p.getTotalLength();
            p.style.strokeDasharray = length;
            p.style.strokeDashoffset = length;
        }
        return p;
    }

    reset() {
        if (this.timeline) clearTimeout(this.timeline);
        this.build();
    }

    play() {
        this.reset();
        let t = 0;
        const step = (fn, delay) => { t += delay; this.timeline = setTimeout(fn, t); };

        step(() => {
            this.elements.edges.forEach(e => {
                e.style.opacity = 0.8;
                e.style.transition = "stroke-dashoffset 1s ease-in-out";
                e.style.strokeDashoffset = "0";
            });
        }, 100);

        step(() => {
            this.elements.repNodes.forEach(n => {
                n.rect.style.opacity = 1;
                n.text.style.opacity = 1;
            });
            this.elements.edges.forEach(e => e.style.opacity = 0);
        }, 1200);

        step(() => {
            this.elements.res.forEach(n => {
                n.rect.style.opacity = 1;
                n.text.style.opacity = 1;
            });
        }, 800);

        step(() => {
            this.elements.jacEdges.forEach(e => {
                e.style.opacity = 0.8;
                e.style.transition = "stroke-dashoffset 1s ease-in-out";
                e.style.strokeDashoffset = "0";
            });
        }, 800);

        step(() => {
            this.elements.jacBlocks.forEach(b => b.block.style.opacity = b.opacity);
        }, 1200);

        step(() => {
            this.elements.jacEdges.forEach(e => e.style.opacity = 0);
        }, 800);
    }
}
