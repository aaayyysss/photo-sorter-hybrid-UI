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
      // ensure shape
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
      // expected: {person_id, vectors: [[...],[...]]}
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

  // Sort
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
      const base = JSON.parse(txt); // should have items[]
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
      // render short table
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

      // Prepare decisions.json to download (client will apply locally)
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
});
