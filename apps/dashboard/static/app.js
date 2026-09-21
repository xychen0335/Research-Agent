async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function metricCard(title, run) {
  if (!run || run.unrun || !run.metrics) {
    const note = run && run.note ? run.note : "";
    return `<article class="card"><h3>${title}</h3><p class="unrun">未运行</p><p class="mute">${run ? run.run_id : ""}</p><p class="mute">${note}</p></article>`;
  }
  const m = run.metrics;
  const em = m.answer_em == null ? "未运行" : Number(m.answer_em).toFixed(3);
  const support = m.evidence_support == null ? "无 gold evidence 标注" : Number(m.evidence_support).toFixed(3);
  const failures = m.failures || {};
  const reasons = failures.reasons ? Object.entries(failures.reasons).map(([k, v]) => `${k}:${v}`).join(" · ") : "";
  return `<article class="card">
    <h3>${title}: ${run.run_id}</h3>
    <p>policy ${run.policy_version || "unknown"} · ${run.live ? "在线运行" : "历史回放"}</p>
    <p>answer EM ${em}</p>
    <p>legal citations ${Number(m.legal_citation_rate || 0).toFixed(3)}</p>
    <p>evidence support ${support}</p>
    <p>mean explore calls ${Number(m.mean_explore_calls || 0).toFixed(2)}</p>
    <p>submit ${Number(m.submit_rate || 0).toFixed(2)} · tool success ${Number((failures.tool_success_rate) || 0).toFixed(2)}</p>
    <p class="mute">${reasons || "无失败明细"}</p>
    <p>episodes ${run.n_episodes}</p>
  </article>`;
}

function cell(item) {
  if (!item || item.unrun) return `<div class="unrun">未运行</div>`;
  return `<div>
    <div>${item.answer || "(empty)"}</div>
    <div class="mute">${item.status} · ${item.policy_version || ""}</div>
  </div>`;
}

async function load() {
  const payload = await fetchJSON("/api/runs");
  const runs = payload.runs;
  const leftSel = document.getElementById("left-run");
  const rightSel = document.getElementById("right-run");
  const options = runs.map((run) => `<option value="${run.run_id}">${run.run_id}${run.unrun ? " (未运行)" : ""}</option>`).join("");
  const previousLeft = leftSel.value;
  const previousRight = rightSel.value;
  leftSel.innerHTML = options;
  rightSel.innerHTML = options;
  if (payload.preferred_left) leftSel.value = payload.preferred_left;
  if (payload.preferred_right) rightSel.value = payload.preferred_right;
  if ([...leftSel.options].some((opt) => opt.value === previousLeft)) leftSel.value = previousLeft;
  if ([...rightSel.options].some((opt) => opt.value === previousRight)) rightSel.value = previousRight;
  if (!runs.length) {
    document.getElementById("metrics").innerHTML = `<article class="card"><p class="unrun">outputs/ 下没有 run</p></article>`;
    document.getElementById("compare").innerHTML = "";
    return;
  }
  const left = leftSel.value;
  const right = rightSel.value;
  const leftRun = await fetchJSON(`/api/runs/${left}`);
  const rightRun = await fetchJSON(`/api/runs/${right}`);
  document.getElementById("metrics").innerHTML = metricCard("左", leftRun) + metricCard("右", rightRun);
  const compared = await fetchJSON(`/api/compare?left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}`);
  document.getElementById("compare").innerHTML = compared.rows.map((row) => `
    <div class="row" role="button" tabindex="0" data-left="${left}" data-right="${right}" data-ltask="${row.left && row.left.episode_id || ""}" data-rtask="${row.right && row.right.episode_id || ""}">
      <div>
        <strong>${row.task_id || ""}</strong>
        <div class="mute">${row.question || ""}</div>
      </div>
      ${cell(row.left)}
      ${cell(row.right)}
    </div>
  `).join("");
  try {
    const show = await fetchJSON("/api/showcase");
    document.getElementById("showcase").innerHTML = `
      <article class="card">
        <h3>${show.task_id}</h3>
        <p>${show.question || ""}</p>
        <p class="mute">${show.note || ""}</p>
        <p><strong>${show.left_run}</strong> ${show.left && show.left.unrun ? "未运行" : (show.left && show.left.answer) || ""}</p>
        <p><strong>${show.right_run}</strong> ${show.right && show.right.unrun ? "未运行" : (show.right && show.right.answer) || ""}</p>
        <div class="planning">${(show.planning && show.planning.conditions || []).map((item) => {
          const plan = item.plan;
          if (item.status !== "ran" || !plan) {
            return `<article class="card"><h4>${item.label || item.name}</h4><p class="unrun">未运行</p><p class="mute">${item.note || ""}</p></article>`;
          }
          return `<article class="card">
            <h4>${item.label || item.name}</h4>
            <p>${plan.reproduced_answer || ""}</p>
            <p class="mute">settings: ${(plan.settings_to_match || []).join(" · ") || "无"}</p>
            <p class="mute">missing: ${(plan.missing_conditions || []).join(" · ") || "无"}</p>
            <p class="mute">human scores: 空（盲评未填）</p>
          </article>`;
        }).join("")}</div>
        <p class="mute">${(show.planning && show.planning.note) || ""}</p>
      </article>`;
  } catch (err) {
    document.getElementById("showcase").innerHTML = `<p class="unrun">案例未加载</p>`;
  }
  for (const node of document.querySelectorAll(".row")) {
    node.addEventListener("click", async () => {
      const episodeId = node.dataset.ltask || node.dataset.rtask;
      const runId = node.dataset.ltask ? node.dataset.left : node.dataset.right;
      if (!episodeId) {
        document.getElementById("detail").textContent = "未运行";
        return;
      }
      const episode = await fetchJSON(`/api/runs/${runId}/episodes/${episodeId}`);
      document.getElementById("detail").textContent = JSON.stringify(episode, null, 2);
    });
  }
}

document.getElementById("reload").addEventListener("click", load);
document.getElementById("left-run").addEventListener("change", load);
document.getElementById("right-run").addEventListener("change", load);
document.getElementById("live-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.getElementById("live-q").value.trim();
  await fetchJSON("/api/live", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      run_id: "live",
      policy: document.getElementById("live-policy").value || "scripted",
    }),
  });
  await load();
});
load().catch((err) => {
  document.getElementById("detail").textContent = String(err);
});
