/**
 * RegGraph Dashboard — Interactive Graph Q&A, Cytoscape Visualization, & Circular Management
 */
document.addEventListener("DOMContentLoaded", () => {
    // State
    let cy = null;
    let selectedArchetype = "AUTO";
    let sampleQueriesData = [];
    let allCirculars = [];
    let targetDeleteDocId = null;
    let targetDeleteRefNum = null;

    // View & Tab Elements
    const tabBtnQa = document.getElementById("tab-btn-qa");
    const tabBtnCirculars = document.getElementById("tab-btn-circulars");
    const viewQa = document.getElementById("view-qa");
    const viewCirculars = document.getElementById("view-circulars");
    const circularsCountBadge = document.getElementById("circulars-count-badge");
    const toastContainer = document.getElementById("toast-container");

    // Q&A DOM Elements
    const queryInput = document.getElementById("query-input");
    const btnExecute = document.getElementById("btn-execute-query");
    const archetypeChips = document.getElementById("archetype-chips");
    const samplePillsContainer = document.getElementById("sample-pills");
    const loader = document.getElementById("query-loader");
    const resultsDashboard = document.getElementById("results-dashboard");
    const planBanner = document.getElementById("plan-banner");
    const planHeader = document.getElementById("toggle-plan");
    const planBody = document.getElementById("plan-body");
    const planTypeBadge = document.getElementById("plan-type");
    const planReasoningText = document.getElementById("plan-reasoning");
    const cypherCodeBlock = document.getElementById("cypher-code");
    const btnCopyCypher = document.getElementById("btn-copy-cypher");

    // Output Cards Elements
    const currentSummary = document.getElementById("current-summary");
    const currentCircular = document.getElementById("current-circular");
    const currentEffectiveDate = document.getElementById("current-effective-date");
    const currentEntities = document.getElementById("current-entities");
    const timelineList = document.getElementById("timeline-list");
    const changesCountBadge = document.getElementById("changes-count");
    const synthesisText = document.getElementById("synthesis-text");
    const graphStats = document.getElementById("graph-stats");

    // Cytoscape Canvas & Inspector
    const cyContainer = document.getElementById("cy-canvas");
    const nodeInspector = document.getElementById("node-inspector");
    const inspectorTitle = document.getElementById("inspector-title");
    const inspectorBody = document.getElementById("inspector-body");
    const btnCloseInspector = document.getElementById("btn-close-inspector");
    const btnFitGraph = document.getElementById("btn-fit-graph");
    const btnRelayoutGraph = document.getElementById("btn-relayout-graph");

    // Circulars Management Elements
    const circularSearchInput = document.getElementById("circular-search-input");
    const circularStatusFilter = document.getElementById("circular-status-filter");
    const btnRefreshCirculars = document.getElementById("btn-refresh-circulars");
    const btnUploadFromCirculars = document.getElementById("btn-upload-from-circulars");
    const circularsContainer = document.getElementById("circulars-container");
    const circularsLoading = document.getElementById("circulars-loading");
    const circularsEmpty = document.getElementById("circulars-empty");

    // Upload Modal Elements
    const uploadModal = document.getElementById("upload-modal");
    const btnOpenUploadModal = document.getElementById("btn-upload-modal");
    const btnCancelUpload = document.getElementById("btn-cancel-upload");
    const btnSubmitUpload = document.getElementById("btn-submit-upload");
    const pdfFileInput = document.getElementById("pdf-file-input");
    const dropZone = document.getElementById("drop-zone");
    const uploadStatusBox = document.getElementById("upload-status-box");
    const uploadProgressText = document.getElementById("upload-progress-text");
    const uploadProgressBar = document.getElementById("upload-progress-bar");

    // Delete Modal Elements
    const deleteModal = document.getElementById("delete-modal");
    const deleteTargetRef = document.getElementById("delete-target-ref");
    const btnCloseDeleteModal = document.getElementById("btn-close-delete-modal");
    const btnCancelDelete = document.getElementById("btn-cancel-delete");
    const btnConfirmDelete = document.getElementById("btn-confirm-delete");

    // ==================== TOAST NOTIFICATIONS ====================
    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        
        let icon = "ℹ️";
        if (type === "success") icon = "✅";
        else if (type === "error") icon = "❌";
        else if (type === "warning") icon = "⚠️";

        toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(-10px)";
            toast.style.transition = "all 0.3s ease";
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    // ==================== TAB SWITCHING ====================
    function switchTab(viewName) {
        if (viewName === "view-qa") {
            tabBtnQa.classList.add("active");
            tabBtnCirculars.classList.remove("active");
            viewQa.classList.remove("hidden");
            viewCirculars.classList.add("hidden");
            if (cy) {
                setTimeout(() => cy.resize(), 100);
            }
        } else if (viewName === "view-circulars") {
            tabBtnCirculars.classList.add("active");
            tabBtnQa.classList.remove("active");
            viewCirculars.classList.remove("hidden");
            viewQa.classList.add("hidden");
            loadCirculars();
        }
    }

    tabBtnQa.addEventListener("click", () => switchTab("view-qa"));
    tabBtnCirculars.addEventListener("click", () => switchTab("view-circulars"));

    // ==================== CYTOSCAPE GRAPH INITIALIZATION ====================
    function initCytoscape() {
        cy = cytoscape({
            container: cyContainer,
            boxSelectionEnabled: false,
            autounselectify: false,
            style: [
                {
                    selector: "node",
                    style: {
                        "label": "data(label)",
                        "color": "#f8fafc",
                        "font-size": "11px",
                        "font-family": "Outfit, sans-serif",
                        "text-valign": "center",
                        "text-halign": "center",
                        "text-wrap": "wrap",
                        "text-max-width": "120px",
                        "text-background-color": "rgba(15, 23, 42, 0.8)",
                        "text-background-opacity": 0.8,
                        "text-background-padding": "3px",
                        "text-background-shape": "roundrectangle",
                        "border-width": 2,
                        "border-color": "#ffffff",
                        "border-opacity": 0.4,
                        "transition-property": "background-color, border-color, width, height",
                        "transition-duration": "0.3s"
                    }
                },
                {
                    selector: 'node[type = "Document"]',
                    style: {
                        "background-color": "#0284c7",
                        "shape": "roundrectangle",
                        "width": 60,
                        "height": 40,
                        "border-color": "#38bdf8"
                    }
                },
                {
                    selector: 'node[type = "Fact"]',
                    style: {
                        "background-color": "#10b981",
                        "shape": "ellipse",
                        "width": 45,
                        "height": 45,
                        "border-color": "#34d399"
                    }
                },
                {
                    selector: 'node[status = "DELETED"]',
                    style: {
                        "background-color": "#ef4444",
                        "border-color": "#f87171"
                    }
                },
                {
                    selector: 'node[status = "MODIFIED"]',
                    style: {
                        "background-color": "#f59e0b",
                        "border-color": "#fbbf24"
                    }
                },
                {
                    selector: 'node[status = "SUPERSEDED"]',
                    style: {
                        "background-color": "#818cf8",
                        "border-color": "#a5b4fc"
                    }
                },
                {
                    selector: 'node[type = "Entity"]',
                    style: {
                        "background-color": "#6366f1",
                        "shape": "hexagon",
                        "width": 45,
                        "height": 45,
                        "border-color": "#c7d2fe"
                    }
                },
                {
                    selector: 'node[type = "ChangeEvent"]',
                    style: {
                        "background-color": "#ec4899",
                        "shape": "diamond",
                        "width": 40,
                        "height": 40,
                        "border-color": "#f472b6"
                    }
                },
                {
                    selector: "edge",
                    style: {
                        "width": 2,
                        "line-color": "rgba(255, 255, 255, 0.2)",
                        "target-arrow-color": "rgba(255, 255, 255, 0.4)",
                        "target-arrow-shape": "triangle",
                        "curve-style": "bezier",
                        "label": "data(label)",
                        "font-size": "9px",
                        "color": "#94a3b8",
                        "text-rotation": "autorotate",
                        "text-background-color": "#0f172a",
                        "text-background-opacity": 0.8,
                        "text-background-padding": "2px"
                    }
                },
                {
                    selector: 'edge[type = "MODIFIES"], edge[type = "DELETES"], edge[type = "SUPERSEDES"]',
                    style: {
                        "width": 3,
                        "line-color": "#f59e0b",
                        "target-arrow-color": "#f59e0b",
                        "line-style": "dashed"
                    }
                },
                {
                    selector: 'edge[type = "AMENDS"]',
                    style: {
                        "width": 3,
                        "line-color": "#38bdf8",
                        "target-arrow-color": "#38bdf8"
                    }
                }
            ],
            layout: {
                name: "cose",
                animate: true,
                padding: 30
            }
        });

        // Node click inspection
        cy.on("tap", "node", (evt) => {
            const node = evt.target;
            displayNodeDetails(node.data());
        });

        // Background click closes inspector
        cy.on("tap", (evt) => {
            if (evt.target === cy) {
                nodeInspector.classList.add("hidden");
            }
        });
    }

    function displayNodeDetails(data) {
        inspectorTitle.textContent = `${data.type || "Node"}: ${data.label || data.id}`;
        
        let html = `<div style="display:flex; flex-direction:column; gap:6px;">`;
        if (data.status) {
            html += `<div><strong>Status:</strong> <span class="badge badge-accent">${data.status}</span></div>`;
        }
        if (data.properties) {
            for (const [k, v] of Object.entries(data.properties)) {
                if (v && typeof v === "string") {
                    html += `<div><strong>${k}:</strong> ${v}</div>`;
                }
            }
        }
        html += `</div>`;
        
        inspectorBody.innerHTML = html;
        nodeInspector.classList.remove("hidden");
    }

    // ==================== INGESTED CIRCULARS MANAGEMENT ====================
    async function loadCirculars() {
        circularsLoading.classList.remove("hidden");
        circularsContainer.innerHTML = "";
        circularsEmpty.classList.add("hidden");

        try {
            const res = await fetch("/api/documents");
            if (!res.ok) throw new Error("Failed to fetch documents.");
            allCirculars = await res.json();
            
            circularsCountBadge.textContent = allCirculars.length;
            renderCircularsList();
        } catch (err) {
            showToast(`Could not load circulars: ${err.message}`, "error");
        } finally {
            circularsLoading.classList.add("hidden");
        }
    }

    function renderCircularsList() {
        circularsContainer.innerHTML = "";
        const queryTerm = (circularSearchInput.value || "").toLowerCase().trim();
        const filterStatus = circularStatusFilter.value;

        const filtered = allCirculars.filter(doc => {
            const matchesQuery = !queryTerm || 
                (doc.reference_number && doc.reference_number.toLowerCase().includes(queryTerm)) ||
                (doc.title && doc.title.toLowerCase().includes(queryTerm)) ||
                (doc.circular_number && doc.circular_number.toLowerCase().includes(queryTerm)) ||
                (doc.filename && doc.filename.toLowerCase().includes(queryTerm)) ||
                (doc.document_date && doc.document_date.toLowerCase().includes(queryTerm));

            const matchesStatus = filterStatus === "ALL" || doc.status === filterStatus;
            return matchesQuery && matchesStatus;
        });

        if (filtered.length === 0) {
            circularsEmpty.classList.remove("hidden");
            return;
        }

        circularsEmpty.classList.add("hidden");

        filtered.forEach(doc => {
            const card = document.createElement("div");
            card.className = "circular-card";

            const refNum = doc.reference_number || doc.circular_number || doc.filename || "Circular";
            const dateStr = doc.document_date || "Date Unknown";
            const sourceStr = doc.source || "REGULATORY";

            let badgeClass = "badge-success";
            if (doc.status === "PROCESSING") badgeClass = "badge-warning";
            else if (doc.status === "FAILED") badgeClass = "badge-danger";

            card.innerHTML = `
                <div class="circular-card-header">
                    <div>
                        <div class="circular-ref-title">📄 ${refNum}</div>
                        ${doc.circular_number && doc.circular_number !== refNum ? `<div style="font-size:12px; color:var(--text-dim);">Circular: ${doc.circular_number}</div>` : ''}
                    </div>
                    <span class="badge ${badgeClass}">${doc.status}</span>
                </div>
                <div class="circular-card-body">
                    <div class="circular-title-text">${doc.title || doc.filename}</div>
                    <div class="circular-meta-row">
                        <span>🏛️ ${sourceStr}</span>
                        <span>📅 ${dateStr}</span>
                        <span>📄 ${doc.total_pages || 1} Page(s)</span>
                    </div>
                    <div class="circular-stats-pills">
                        <span class="stat-pill">Sections: ${doc.sections_created || 0}</span>
                        <span class="stat-pill">Chunks: ${doc.chunks_created || 0}</span>
                        <span class="stat-pill">Facts: ${doc.facts_extracted || 0}</span>
                    </div>
                </div>
                <div class="circular-card-footer">
                    <div class="card-actions-group">
                        <button class="btn btn-secondary btn-sm btn-query-doc" data-ref="${refNum}">
                            🔍 Q&A
                        </button>
                        <button class="btn btn-secondary btn-sm btn-lineage-doc" data-ref="${refNum}">
                            🔄 Lineage
                        </button>
                    </div>
                    <button class="btn btn-danger-outline btn-sm btn-delete-doc" data-id="${doc.document_id}" data-ref="${refNum}">
                        🗑️ Delete
                    </button>
                </div>
            `;

            // Action: Query this circular
            card.querySelector(".btn-query-doc").addEventListener("click", () => {
                queryInput.value = `What are the active requirements in ${refNum}?`;
                switchTab("view-qa");
                executeQuery();
            });

            // Action: Lineage for this circular
            card.querySelector(".btn-lineage-doc").addEventListener("click", () => {
                queryInput.value = `Show the evolution and amendments of ${refNum}`;
                selectedArchetype = "EVOLUTION";
                document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
                document.querySelector('.chip[data-type="EVOLUTION"]').classList.add("active");
                switchTab("view-qa");
                executeQuery();
            });

            // Action: Delete prompt
            card.querySelector(".btn-delete-doc").addEventListener("click", () => {
                targetDeleteDocId = doc.document_id;
                targetDeleteRefNum = refNum;
                deleteTargetRef.textContent = refNum;
                deleteModal.classList.remove("hidden");
            });

            circularsContainer.appendChild(card);
        });
    }

    // Filter listeners
    circularSearchInput.addEventListener("input", renderCircularsList);
    circularStatusFilter.addEventListener("change", renderCircularsList);
    btnRefreshCirculars.addEventListener("click", loadCirculars);
    btnUploadFromCirculars.addEventListener("click", () => uploadModal.classList.remove("hidden"));

    // ==================== DELETE MODAL HANDLERS ====================
    btnCloseDeleteModal.addEventListener("click", () => deleteModal.classList.add("hidden"));
    btnCancelDelete.addEventListener("click", () => deleteModal.classList.add("hidden"));

    btnConfirmDelete.addEventListener("click", async () => {
        if (!targetDeleteDocId) return;

        btnConfirmDelete.setAttribute("disabled", "true");
        btnConfirmDelete.innerHTML = `<span>Deleting from Neo4j...</span>`;

        try {
            const res = await fetch(`/api/documents/${targetDeleteDocId}`, {
                method: "DELETE"
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Delete failed");
            }

            deleteModal.classList.add("hidden");
            showToast(`Circular ${targetDeleteRefNum} successfully deleted from Neo4j graph!`, "success");
            
            // Refresh list
            loadCirculars();
        } catch (err) {
            showToast(`Failed to delete circular: ${err.message}`, "error");
        } finally {
            btnConfirmDelete.removeAttribute("disabled");
            btnConfirmDelete.innerHTML = `
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                Yes, Delete from Graph
            `;
        }
    });

    // ==================== QUERY ARCHETYPES & SAMPLE QUESTIONS ====================
    async function loadArchetypes() {
        try {
            const res = await fetch("/api/query/archetypes");
            if (!res.ok) return;
            sampleQueriesData = await res.json();
            renderSamplePills();
        } catch (e) {
            console.warn("Could not load query archetypes:", e);
        }
    }

    function renderSamplePills() {
        samplePillsContainer.innerHTML = "";
        let queries = [];
        if (selectedArchetype === "AUTO") {
            sampleQueriesData.forEach(g => {
                if (g.sample_queries && g.sample_queries.length > 0) {
                    queries.push({ text: g.sample_queries[0], type: g.type });
                }
            });
        } else {
            const group = sampleQueriesData.find(g => g.type === selectedArchetype);
            if (group && group.sample_queries) {
                queries = group.sample_queries.map(q => ({ text: q, type: group.type }));
            }
        }

        queries.slice(0, 4).forEach(item => {
            const pill = document.createElement("button");
            pill.className = "sample-pill";
            pill.textContent = item.text;
            pill.addEventListener("click", () => {
                queryInput.value = item.text;
                executeQuery();
            });
            samplePillsContainer.appendChild(pill);
        });
    }

    // Archetype chip selection
    archetypeChips.addEventListener("click", (e) => {
        if (!e.target.classList.contains("chip")) return;
        document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
        e.target.classList.add("active");
        selectedArchetype = e.target.getAttribute("data-type");
        renderSamplePills();
    });

    // Toggle Plan & Cypher View
    planHeader.addEventListener("click", () => {
        planBody.classList.toggle("hidden");
        const chevron = planHeader.querySelector(".chevron");
        if (chevron) {
            chevron.style.transform = planBody.classList.contains("hidden") ? "rotate(0deg)" : "rotate(180deg)";
        }
    });

    // Copy Cypher Query
    btnCopyCypher.addEventListener("click", () => {
        const code = cypherCodeBlock.textContent;
        navigator.clipboard.writeText(code).then(() => {
            btnCopyCypher.textContent = "Copied!";
            setTimeout(() => { btnCopyCypher.textContent = "Copy"; }, 2000);
        });
    });

    // ==================== EXECUTE REGULATORY QUERY ====================
    async function executeQuery() {
        const queryText = queryInput.value.trim();
        if (!queryText) return;

        loader.classList.remove("hidden");
        resultsDashboard.classList.add("hidden");
        planBanner.classList.add("hidden");
        nodeInspector.classList.add("hidden");

        const payload = {
            query: queryText,
            query_type_override: selectedArchetype === "AUTO" ? null : selectedArchetype,
            include_subgraph: true
        };

        try {
            const response = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Query execution failed.");
            }

            const data = await response.json();
            renderResults(data);
        } catch (err) {
            showToast(`Query failed: ${err.message}`, "error");
        } finally {
            loader.classList.add("hidden");
        }
    }

    function renderResults(data) {
        // 1. Plan & Cypher banner
        planTypeBadge.textContent = data.plan.query_type;
        planReasoningText.textContent = data.plan.reasoning;
        cypherCodeBlock.textContent = data.cypher_query;
        planBanner.classList.remove("hidden");

        // 2. Part 1: Current Regulation Summary Card
        if (data.current_regulation) {
            currentSummary.textContent = data.current_regulation.summary;
            currentCircular.textContent = data.current_regulation.primary_circular || "N/A";
            currentEffectiveDate.textContent = data.current_regulation.effective_date || "N/A";
            currentEntities.textContent = (data.current_regulation.entities_affected || []).join(", ") || "All Regulated Entities";
        } else {
            currentSummary.textContent = "No currently active requirement identified for this specific query scope.";
            currentCircular.textContent = "—";
            currentEffectiveDate.textContent = "—";
            currentEntities.textContent = "—";
        }

        // 3. Part 2: How It Changed (Chronological Delta Timeline)
        timelineList.innerHTML = "";
        const changes = data.change_history || [];
        changesCountBadge.textContent = `${changes.length} Change${changes.length === 1 ? '' : 's'}`;

        if (changes.length === 0) {
            timelineList.innerHTML = `<p style="font-size:13px; color:var(--text-dim); padding:10px 0;">No chronological amendments or deletion events detected for this query scope.</p>`;
        } else {
            changes.forEach(c => {
                const item = document.createElement("div");
                item.className = "timeline-item";
                
                let markerClass = "marker-modified";
                if (c.change_type === "DELETED") markerClass = "marker-deleted";
                else if (c.change_type === "SUPERSEDED") markerClass = "marker-superseded";

                item.innerHTML = `
                    <div class="timeline-marker ${markerClass}"></div>
                    <div class="timeline-content">
                        <div class="timeline-header">
                            <span class="badge ${c.change_type === 'DELETED' ? 'badge-danger' : 'badge-warning'}">${c.change_type}</span>
                            <span class="timeline-date">${c.date || 'Effective Date N/A'}</span>
                        </div>
                        <p class="timeline-desc">${c.description}</p>
                        ${c.amending_circular ? `<div style="margin-top:6px; font-size:11px; color:var(--text-dim);">Amending Notice: <code>${c.amending_circular}</code></div>` : ''}
                    </div>
                `;
                timelineList.appendChild(item);
            });
        }

        // 4. Synthesized Answer & Citations
        if (window.marked) {
            synthesisText.innerHTML = marked.parse(data.llm_answer || "");
        } else {
            synthesisText.textContent = data.llm_answer || "";
        }

        // 5. Part 3: Cytoscape Graph Subgraph
        renderSubgraph(data.subgraph);

        resultsDashboard.classList.remove("hidden");
    }

    function renderSubgraph(subgraph) {
        if (!cy) initCytoscape();

        cy.elements().remove();

        if (!subgraph || !subgraph.nodes || subgraph.nodes.length === 0) {
            graphStats.textContent = "0 nodes, 0 edges";
            return;
        }

        const cyElements = [];

        subgraph.nodes.forEach(n => {
            cyElements.push({
                group: "nodes",
                data: {
                    id: n.id,
                    label: n.label,
                    type: n.type,
                    status: n.status,
                    properties: n.properties
                }
            });
        });

        subgraph.edges.forEach(e => {
            cyElements.push({
                group: "edges",
                data: {
                    id: e.id,
                    source: e.source,
                    target: e.target,
                    label: e.type,
                    type: e.type,
                    properties: e.properties
                }
            });
        });

        cy.add(cyElements);
        graphStats.textContent = `${subgraph.total_nodes} nodes, ${subgraph.total_edges} edges`;

        // Layout
        cy.layout({
            name: "cose",
            animate: true,
            padding: 40,
            nodeRepulsion: 6500,
            idealEdgeLength: 80
        }).run();
    }

    // Graph Toolbar actions
    btnFitGraph.addEventListener("click", () => {
        if (cy) cy.fit(null, 30);
    });

    btnRelayoutGraph.addEventListener("click", () => {
        if (cy) {
            cy.layout({ name: "cose", animate: true, padding: 40 }).run();
        }
    });

    btnCloseInspector.addEventListener("click", () => {
        nodeInspector.classList.add("hidden");
    });

    // Execute on Enter / Click
    btnExecute.addEventListener("click", executeQuery);
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") executeQuery();
    });

    // ==================== UPLOAD PDF MODAL ====================
    document.querySelectorAll(".btn-open-upload-modal-trigger, #btn-upload-modal").forEach(btn => {
        btn.addEventListener("click", () => uploadModal.classList.remove("hidden"));
    });

    document.querySelectorAll(".btn-close-upload-modal, #btn-cancel-upload").forEach(btn => {
        btn.addEventListener("click", () => uploadModal.classList.add("hidden"));
    });

    pdfFileInput.addEventListener("change", () => {
        if (pdfFileInput.files.length > 0) {
            const file = pdfFileInput.files[0];
            dropZone.querySelector(".drop-prompt").innerHTML = `Selected: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)`;
            btnSubmitUpload.removeAttribute("disabled");
        }
    });

    btnSubmitUpload.addEventListener("click", async () => {
        if (pdfFileInput.files.length === 0) return;
        const file = pdfFileInput.files[0];

        const formData = new FormData();
        formData.append("file", file);

        uploadStatusBox.classList.remove("hidden");
        uploadProgressText.textContent = "Uploading PDF to server...";
        uploadProgressBar.style.width = "30%";
        btnSubmitUpload.setAttribute("disabled", "true");

        try {
            const uploadRes = await fetch("/api/documents/upload", {
                method: "POST",
                body: formData
            });

            if (!uploadRes.ok) {
                const err = await uploadRes.json();
                throw new Error(err.detail || "Upload failed");
            }

            const uploadData = await uploadRes.json();
            const docId = uploadData.document_id;
            uploadProgressText.textContent = `Pipeline active: Extracting hierarchy & atomic facts...`;
            uploadProgressBar.style.width = "65%";

            // Poll status
            const interval = setInterval(async () => {
                try {
                    const statusRes = await fetch(`/api/documents/${docId}/status`);
                    if (statusRes.ok) {
                        const statusData = await statusRes.json();
                        uploadProgressText.textContent = `Stage: ${statusData.current_stage} (${statusData.facts_extracted || 0} facts extracted)`;
                        if (statusData.status === "COMPLETED") {
                            clearInterval(interval);
                            uploadProgressBar.style.width = "100%";
                            uploadProgressText.textContent = `✅ Successfully ingested into Neo4j graph!`;
                            showToast(`Circular "${file.name}" ingested successfully!`, "success");
                            setTimeout(() => {
                                uploadModal.classList.add("hidden");
                                uploadStatusBox.classList.add("hidden");
                                pdfFileInput.value = "";
                                dropZone.querySelector(".drop-prompt").innerHTML = `Drag & Drop Circular PDF here or <strong>Browse</strong>`;
                                loadCirculars();
                                queryInput.value = `What are the active requirements in ${file.name}?`;
                                executeQuery();
                            }, 1500);
                        } else if (statusData.status === "FAILED") {
                            clearInterval(interval);
                            uploadProgressText.textContent = `❌ Ingestion failed: ${statusData.errors.join(', ')}`;
                            showToast(`Ingestion failed: ${statusData.errors.join(', ')}`, "error");
                        }
                    }
                } catch (e) {
                    console.error("Polling error:", e);
                }
            }, 1000);

        } catch (e) {
            uploadProgressText.textContent = `❌ Error: ${e.message}`;
            btnSubmitUpload.removeAttribute("disabled");
            showToast(`Upload failed: ${e.message}`, "error");
        }
    });

    // Initial setup
    initCytoscape();
    loadArchetypes();
    loadCirculars();
});
