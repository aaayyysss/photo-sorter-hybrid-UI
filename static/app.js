async function getJSON(url, opts = {}) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = "";
    try { msg = (await r.json()).message || r.statusText; } catch {}
    throw new Error(msg || `HTTP ${r.status}`);
  }
  return r.json();
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsText(file, "utf-8");
  });
}

function downloadJSON(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderPeople(list, el) {
  el.innerHTML = "";
  if (!list || !list.length) {
    el.innerHTML = "<div class='muted'>No references yet.</div>";
    return;
  }
  const table = document.createElement("table");
  table.innerHTML = `
    <thead><tr>
      <th>Person</th><th>#Vec</th><th>Dims</th><th>μ (pairs)</th><th>σ</th>
    </tr></thead><tbody></tbody>`;
  const tb = table.querySelector("tbody");
  list.forEach(p => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${p.person_id}</td>
      <td>${p.n_vectors}</td>
      <td>${p.dims}</td>
      <td>${(p.mu_pairs||0).toFixed(3)}</td>
      <td>${(p.sigma_pairs||0).toFixed(3)}</td>`;
    tb.appendChild(tr);
  });
  el.appendChild(table);
}

async function refreshHealth() {
  const h = document.getElementById("health");
  const refsList = document.getElementById("refsList");
  try {
    const data = await getJSON("/api/health");
    h.textContent = `✅ Server OK — dims=${data.dims || "?"}, persons=${(data.people||[]).length}`;
    renderPeople(data.people || [], refsList);
  } catch (e) {
    h.textContent = `❌ Server error: ${e.message}`;
  }
}

async function postJSON(url, body, adminToken) {
  const headers = { "Content-Type": "application/json" };
  if (adminToken) headers["X-Admin-Token"] = adminToken;
  const res = await fetch(url, { method: "POST", headers, body: JSON.stringify(body) });
  if (!res.ok) {
    let msg = "";
    try { msg = (await res.json()).message || res.statusText; } catch {}
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return res.json();
}

// ---------------- Local Companion helpers ----------------
const COMPANION = {
  base: "http://127.0.0.1:8765",
  token: null,
};

async function pingCompanion() {
  try {
    const r = await fetch(`${COMPANION.base}/status`, { method: "GET" });
    if (!r.ok) throw new Error("not ok");
    return r.json();
  } catch (e) {
    return null;
  }
}

async function callCompanion(path, body) {
  const headers = { "Content-Type": "application/json" };
  if (COMPANION.token) headers["X-Companion-Token"] = COMPANION.token;
  const r = await fetch(`${COMPANION.base}${path}`, { method: "POST", headers, body: JSON.stringify(body || {}) });
  if (!r.ok) {
    let msg = "";
    try { msg = (await r.json()).message || r.statusText; } catch {}
    throw new Error(msg || `HTTP ${r.status}`);
  }
  return r.json();
}

window.addEventListener("DOMContentLoaded", () => {
  refreshHealth();

  const refsMsg = document.getElementById("refsMsg");
  const refsList = document.getElementById("refsList");
  const adminEl = document.getElementById("adm");
  const modeEl = document.getElementById("mode");
  const refsFile = document.getElementById("refsJson");

  document.getElementById("sendRefsBatch").onclick = async () => {
    refsMsg.textContent = "Uploading batch…";
    try {
      if (!refsFile.files.length) throw new Error("Pick a refs_register_batch.json first.");
      const txt = await readFileAsText(refsFile.files[0]);
      const obj = JSON.parse(txt);
      if (!Array.isArray(obj.persons)) throw new Error("Invalid format: persons[] missing.");
      obj.mode = modeEl.value || "merge";
      const data = await postJSON("/api/refs/register-batch", obj, adminEl.value.trim());
      refsMsg.textContent = data.message || "Batch registered.";
      renderPeople(data.people || [], refsList);
    } catch (e) {
      refsMsg.textContent = `Error: ${e.message}`;
    }
  };

  document.getElementById("sendRefsSingle").onclick = async () => {
    refsMsg.textContent = "Uploading single…";
    try {
      if (!refsFile.files.length) throw new Error("Pick a single-person JSON first.");
      const txt = await readFileAsText(refsFile.files[0]);
      const obj = JSON.parse(txt);
      obj.mode = modeEl.value || "merge";
      const data = await postJSON("/api/refs/register", obj, adminEl.value.trim());
      refsMsg.textContent = data.message || "Registered.";
      renderPeople(data.people || [], refsList);
    } catch (e) {
      refsMsg.textContent = `Error: ${e.message}`;
    }
  };

  document.getElementById("clearRefs").onclick = async () => {
    if (!confirm("Really clear all references?")) return;
    refsMsg.textContent = "Clearing…";
    try {
      const data = await postJSON("/api/refs/clear", {}, adminEl.value.trim());
      refsMsg.textContent = data.message || "Cleared.";
      renderPeople([], refsList);
    } catch (e) {
      refsMsg.textContent = `Error: ${e.message}`;
    }
  };

  document.getElementById("exportRefs").onclick = () => {
    const token = adminEl.value.trim();
    const url = token ? `/api/refs/export?admin_token=${encodeURIComponent(token)}` : "/api/refs/export";
    window.open(url, "_blank");
  };

  // Sort (manual JSON)
  const sortMsg = document.getElementById("sortMsg");
  const sortResults = document.getElementById("sortResults");
  const inboxJson = document.getElementById("inboxJson");
  const thrEl = document.getElementById("thr");
  const adaptiveEl = document.getElementById("adaptive");
  const akEl = document.getElementById("ak");
  const policyEl = document.getElementById("policy");
  const dlBtn = document.getElementById("downloadDecisions");

  let lastDecision = null;

  document.getElementById("runSort").onclick = async () => {
    sortMsg.textContent = "Running…";
    sortResults.innerHTML = "";
    dlBtn.style.display = "none";
    try {
      if (!inboxJson.files.length) throw new Error("Pick an inbox_embeddings.json file first.");
      const txt = await readFileAsText(inboxJson.files[0]);
      const base = JSON.parse(txt);
      if (!Array.isArray(base.items) || base.items.length === 0) {
        throw new Error("Invalid inbox JSON: items[] missing or empty.");
      }
      const body = {
        items: base.items,
        params: {
          global_threshold_pct: parseInt(thrEl.value || "32", 10),
          adaptive_on: !!adaptiveEl.checked,
          adaptive_k: parseFloat(akEl.value || "1.0"),
          multi_face_policy: policyEl.value || "copy_all",
        }
      };
      const data = await postJSON("/api/sort", body);
      sortMsg.textContent = `Done. Faces: ${data.summary.n_faces}, Images: ${data.summary.n_images}`;
      const tbl = document.createElement("table");
      tbl.innerHTML = `
        <thead><tr>
          <th>Image</th><th>Face</th><th>Person</th><th>Score</th><th>Thr</th><th>Decision</th>
        </tr></thead><tbody></tbody>`;
      const tb = tbl.querySelector("tbody");
      (data.entries || []).slice(0, 2000).forEach(e => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td title="${e.image_id||""}">${(e.image_id||"").toString().split(/[\\/]/).pop()}</td>
          <td>${e.face_id||""}</td>
          <td>${e.best_person||""}</td>
          <td>${typeof e.score === "number" ? e.score.toFixed(3) : ""}</td>
          <td>${typeof e.threshold === "number" ? e.threshold.toFixed(3) : ""}</td>
          <td>${e.decision||""}</td>`;
        tb.appendChild(tr);
      });
      sortResults.appendChild(tbl);

      lastDecision = {
        created_at: new Date().toISOString(),
        params: body.params,
        summary: data.summary,
        entries: data.entries
      };
      dlBtn.style.display = "inline-block";
    } catch (e) {
      sortMsg.textContent = `Error: ${e.message}`;
    }
  };

  dlBtn.onclick = () => {
    if (!lastDecision) return;
    downloadJSON(lastDecision, "decisions.json");
  };

  // ---------- Local Companion UI wiring ----------
  const compStatus = document.getElementById("companionStatus");
  const compOut = document.getElementById("companionOut");
  const tokInput = document.getElementById("compTok");
  const saveTokBtn = document.getElementById("saveTok");
  COMPANION.token = localStorage.getItem("companion_token") || "";
  if (COMPANION.token && tokInput) tokInput.value = COMPANION.token;

  async function refreshCompanion() {
    const st = await pingCompanion();
    if (st) {
      compStatus.textContent = "🔌 Local Companion: connected";
    } else {
      compStatus.innerHTML = "🔌 Local Companion: not detected — run it and paste token. See the README link above.";
      setTimeout(refreshCompanion, 4000);
    }
  }
  if (compStatus) refreshCompanion();

  if (saveTokBtn) {
    saveTokBtn.onclick = () => {
      const v = (tokInput?.value || "").trim();
      if (!v) { alert("Paste the token from the local companion console."); return; }
      localStorage.setItem("companion_token", v);
      COMPANION.token = v;
      compOut.textContent = "Saved token.";
      refreshCompanion();
    };
  }

  const refsPathEl = document.getElementById("refsPath");
  const inboxPathEl = document.getElementById("inboxPath");
  const sortedPathEl = document.getElementById("sortedPath");
  const companionModeEl = document.getElementById("companionMode");
  const applyModeEl = document.getElementById("applyMode");

  const btnBuildRefs = document.getElementById("btnBuildAndRegister");
  const btnScanSort = document.getElementById("btnScanAndSort");
  const btnApply = document.getElementById("btnApplyDecisions");

  if (btnBuildRefs) btnBuildRefs.onclick = async () => {
    try {
      compOut.textContent = "Computing refs locally…";
      const body = {
        refs_path: (refsPathEl?.value || "").trim(),
        mode: (companionModeEl?.value || "merge"),
        det_size: 640
      };
      if (!body.refs_path) throw new Error("Enter a Refs folder path.");
      const resp = await callCompanion("/compute-refs", body);
      compOut.textContent = (resp.message || "Done.") + (resp.server_response ? " (server updated)" : "");
      await refreshHealth();
    } catch (e) {
      compOut.textContent = "Error: " + e.message;
    }
  };

  if (btnScanSort) btnScanSort.onclick = async () => {
    try {
      compOut.textContent = "Scanning inbox & sorting…";
      const body = {
        inbox_path: (inboxPathEl?.value || "").trim(),
        det_size: 640,
        global_threshold_pct: parseInt(document.getElementById("thr").value || "32", 10),
        adaptive_on: !!document.getElementById("adaptive").checked,
        adaptive_k: parseFloat(document.getElementById("ak").value || "1.0"),
        multi_face_policy: document.getElementById("policy").value || "copy_all",
        upload_to_server: true,
      };
      if (!body.inbox_path) throw new Error("Enter an Inbox folder path.");
      const resp = await callCompanion("/compute-inbox", body);
      compOut.textContent = (resp.message || "Done.") + (resp.server_response ? " (results ready)" : "");

      if (resp.server_response?.entries) {
        lastDecision = {
          created_at: new Date().toISOString(),
          params: {
            global_threshold_pct: body.global_threshold_pct,
            adaptive_on: body.adaptive_on,
            adaptive_k: body.adaptive_k,
            multi_face_policy: body.multi_face_policy,
          },
          summary: resp.server_response.summary,
          entries: resp.server_response.entries
        };
        document.getElementById("downloadDecisions").style.display = "inline-block";
        const sortResults = document.getElementById("sortResults");
        const sortMsg = document.getElementById("sortMsg");
        sortResults.innerHTML = "";
        const tbl = document.createElement("table");
        tbl.innerHTML = `
          <thead><tr>
            <th>Image</th><th>Face</th><th>Person</th><th>Score</th><th>Thr</th><th>Decision</th>
          </tr></thead><tbody></tbody>`;
        const tb = tbl.querySelector("tbody");
        (lastDecision.entries || []).slice(0, 2000).forEach(e => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td title="${e.image_id||""}">${(e.image_id||"").toString().split(/[\\/]/).pop()}</td>
            <td>${e.face_id||""}</td>
            <td>${e.best_person||""}</td>
            <td>${typeof e.score === "number" ? e.score.toFixed(3) : ""}</td>
            <td>${typeof e.threshold === "number" ? e.threshold.toFixed(3) : ""}</td>
            <td>${e.decision||""}</td>`;
          tb.appendChild(tr);
        });
        sortResults.appendChild(tbl);
        sortMsg.textContent = `Done. Faces: ${lastDecision.summary.n_faces}, Images: ${lastDecision.summary.n_images}`;
      }
    } catch (e) {
      compOut.textContent = "Error: " + e.message;
    }
  };

  if (btnApply) btnApply.onclick = async () => {
    try {
      if (!lastDecision) { compOut.textContent = "No decisions available. Run Scan & Sort first or upload decisions.json."; return; }
      const body = {
        decisions_json: lastDecision,
        inbox_path: (inboxPathEl?.value || "").trim(),
        sorted_path: (sortedPathEl?.value || "").trim(),
        mode: (applyModeEl?.value || "move")
      };
      if (!body.inbox_path || !body.sorted_path) throw new Error("Enter Inbox and Sorted paths.");
      const resp = await callCompanion("/apply-decisions", body);
      compOut.textContent = `Applied locally. moved=${resp.moved}, copied_or_linked=${resp.copied_or_linked}, skipped=${resp.skipped}`;
    } catch (e) {
      compOut.textContent = "Error: " + e.message;
    }
  };
});
