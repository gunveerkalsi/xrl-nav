"""Self-contained interactive HTML viewer for reviewing explained episodes step by step.

Open the generated file in any browser: no server or internet needed.
"""
from __future__ import annotations

import json

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>XRL Navigation Replay</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d2329;--mute:#6b7280;--line:#e2e2de;--sup:#c0392b;--ag:#2471a3;--acc:#e67e22}
@media (prefers-color-scheme:dark){:root{--bg:#15181c;--card:#1e2227;--ink:#e8eaed;--mute:#9aa0a6;--line:#30353b}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0 12px 0 0}select,button{font:inherit;padding:5px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink);cursor:pointer}
main{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:14px;padding:14px 18px}
@media (max-width:860px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h3{margin:0 0 8px;font-size:13px;color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
canvas{width:100%;height:auto;display:block}
.controls{display:flex;gap:8px;align-items:center;margin-top:10px}input[type=range]{flex:1}
.row{display:flex;align-items:center;gap:6px;margin:3px 0;font-size:12px}.row .lbl{width:92px;text-align:right;color:var(--mute)}
.row .track{flex:1;height:14px;position:relative;background:transparent}.row .bar{position:absolute;top:0;height:14px;border-radius:3px}
.row .val{width:48px;font-variant-numeric:tabular-nums}
#text{font-size:13.5px;background:var(--bg);border-radius:8px;padding:10px;margin-top:6px}
.flags{display:flex;flex-wrap:wrap;gap:6px}.flag{font-size:12px;padding:3px 8px;border-radius:12px;border:1px solid var(--line);cursor:pointer;background:var(--bg)}
.flag:hover{border-color:var(--acc)}.meta{color:var(--mute);font-size:12.5px}
.legend span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:middle}
</style></head><body>
<header><h1>Explainable RL Navigation: decision replay</h1>
<label>Episode <select id="ep"></select></label>
<label>Lidar colour <select id="mode"><option value="ig">Integrated Gradients</option><option value="attention">Attention</option><option value="none">none</option></select></label>
<span class="meta" id="epmeta"></span></header>
<main>
<section class="card"><canvas id="cv" width="640" height="640"></canvas>
<div class="controls"><button id="play">Play</button><button id="prev">&lsaquo;</button><button id="next">&rsaquo;</button>
<input type="range" id="slider" min="0" value="0"><span id="stepl" class="meta"></span></div>
<div class="legend meta"><span style="background:var(--sup)"></span>supports chosen action<span style="background:var(--ag)"></span>argues against<span style="background:#8e44ad"></span>attention</div></section>
<section style="display:flex;flex-direction:column;gap:14px">
<div class="card"><h3>Explanation</h3><div id="text"></div></div>
<div class="card"><h3>Policy &pi;(a|s)</h3><div id="probs"></div></div>
<div class="card"><h3>Group Shapley contribution</h3><div id="shap"></div></div>
<div class="card"><h3>Flagged decisions (click to jump)</h3><div class="flags" id="flags"></div></div>
</section></main>
<script>
const EPISODES = __DATA__;
const $=id=>document.getElementById(id);const cv=$("cv"),ctx=cv.getContext("2d");
let E=null,t=0,timer=null;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
EPISODES.forEach((e,i)=>{const o=document.createElement("option");o.value=i;o.textContent=e.label;$("ep").appendChild(o)});
function load(i){E=EPISODES[i];t=0;$("slider").max=E.steps.length-1;const m=E.metrics;
 $("epmeta").textContent=`${E.layout.name} | ${m.success?"success":m.collision?"collision":"timeout"} | ${m.steps} steps | reward ${m.reward.toFixed(1)}`+(m.path_efficiency?` | path eff ${m.path_efficiency.toFixed(2)}`:"");
 const f=$("flags");f.innerHTML="";if(!E.flags.length)f.innerHTML='<span class="meta">none</span>';
 E.flags.slice(0,60).forEach(fl=>{const b=document.createElement("span");b.className="flag";b.textContent=`t=${fl.t}: ${fl.reasons.join(", ")}`;b.onclick=()=>show(fl.t);f.appendChild(b)});show(0)}
function W(x,y){const h=E.arena_half,s=cv.width/(2*h+0.4);return[(x+h+0.2)*s,cv.height-(y+h+0.2)*s]}
function S(){return cv.width/(2*E.arena_half+0.4)}
function draw(){const st=E.steps[t],s=S();ctx.clearRect(0,0,cv.width,cv.height);ctx.fillStyle=css("--card");ctx.fillRect(0,0,cv.width,cv.height);
 const h=E.arena_half;let [x0,y0]=W(-h,h);ctx.strokeStyle="#555";ctx.lineWidth=2;ctx.strokeRect(x0,y0,2*h*s,2*h*s);
 E.layout.obstacles.forEach(o=>{const [cx,cy]=W(o.pos[0],o.pos[1]);ctx.fillStyle="#cf7d3e";ctx.strokeStyle="#7a4a1a";
  if(o.type==="cylinder"){ctx.beginPath();ctx.arc(cx,cy,o.radius*s,0,7);ctx.fill();ctx.stroke()}
  else{ctx.save();ctx.translate(cx,cy);ctx.rotate(-o.yaw);ctx.fillRect(-o.half[0]*s,-o.half[1]*s,2*o.half[0]*s,2*o.half[1]*s);ctx.strokeRect(-o.half[0]*s,-o.half[1]*s,2*o.half[0]*s,2*o.half[1]*s);ctx.restore()}});
 let [gx,gy]=W(...E.layout.goal);ctx.fillStyle="rgba(46,204,113,.5)";ctx.beginPath();ctx.arc(gx,gy,.35*s,0,7);ctx.fill();
 ctx.strokeStyle="#2e86c1";ctx.lineWidth=2;ctx.beginPath();E.steps.slice(0,t+1).forEach((q,i)=>{const [a,b]=W(q.pose[0],q.pose[1]);i?ctx.lineTo(a,b):ctx.moveTo(a,b)});ctx.stroke();
 const [px,py,th]=st.pose,n=E.n_rays,mode=$("mode").value;let w=null,signed=true;
 if(mode==="ig")w=st.ig.slice(0,n);else if(mode==="attention"&&st.attention){w=st.attention;signed=false}
 const mx=w?Math.max(...w.map(Math.abs))+1e-9:1;const [rx,ry]=W(px,py);
 for(let k=0;k<n;k++){const a=th+k*2*Math.PI/n,L=.2+st.rays[k]*E.max_range;const [ex,ey]=W(px+L*Math.cos(a),py+L*Math.sin(a));
  let c="rgba(150,150,150,.5)",lw=1;if(w){const v=w[k],m=Math.abs(v)/mx;c=signed?(v>0?`rgba(192,57,43,${.2+.8*m})`:`rgba(36,113,163,${.2+.8*m})`):`rgba(142,68,173,${.2+.8*m})`;lw=.6+5*m}
  ctx.strokeStyle=c;ctx.lineWidth=lw;ctx.beginPath();ctx.moveTo(rx,ry);ctx.lineTo(ex,ey);ctx.stroke();if(st.rays[k]<.98){ctx.fillStyle=css("--ink");ctx.fillRect(ex-1.5,ey-1.5,3,3)}}
 ctx.fillStyle="#2e86c1";ctx.beginPath();ctx.arc(rx,ry,.2*s,0,7);ctx.fill();ctx.strokeStyle="#fff";ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(rx,ry);ctx.lineTo(rx+.32*s*Math.cos(th),ry-.32*s*Math.sin(th));ctx.stroke()}
function bars(el,items,signed){const mx=Math.max(...items.map(i=>Math.abs(i[1])))+1e-9;el.innerHTML="";
 items.forEach(([l,v,hi])=>{const r=document.createElement("div");r.className="row";const pct=Math.abs(v)/mx*(signed?50:100);
  const left=signed?(v>=0?50:50-pct):0;const col=signed?(v>=0?css("--sup"):css("--ag")):(hi?css("--acc"):"#8a9099");
  r.innerHTML=`<span class="lbl">${l}</span><span class="track">${signed?'<span style="position:absolute;left:50%;top:0;bottom:0;border-left:1px solid #999"></span>':""}<span class="bar" style="left:${left}%;width:${pct}%;background:${col}"></span></span><span class="val">${v.toFixed(2)}</span>`;el.appendChild(r)})}
function show(i){t=Math.max(0,Math.min(i,E.steps.length-1));$("slider").value=t;const st=E.steps[t];$("stepl").textContent=`step ${t} / ${E.steps.length-1}`;
 draw();$("text").textContent=st.text;
 if(st.probs.length===E.action_names.length&&E.action_names.length>2)bars($("probs"),E.action_names.map((a,k)=>[a,st.probs[k],k===st.action]),false);
 else bars($("probs"),[["v",st.probs[0]],["w",st.probs[1]]],true);
 bars($("shap"),Object.entries(st.shapley).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).map(([k,v])=>[k.replace("lidar:",""),v]),true)}
$("ep").onchange=e=>load(+e.target.value);$("mode").onchange=()=>draw();$("slider").oninput=e=>show(+e.target.value);
$("prev").onclick=()=>show(t-1);$("next").onclick=()=>show(t+1);
$("play").onclick=()=>{if(timer){clearInterval(timer);timer=null;$("play").textContent="Play"}else{$("play").textContent="Pause";timer=setInterval(()=>{if(t>=E.steps.length-1){clearInterval(timer);timer=null;$("play").textContent="Play"}else show(t+1)},120)}};
document.addEventListener("keydown",e=>{if(e.key==="ArrowRight")show(t+1);if(e.key==="ArrowLeft")show(t-1)});
load(0);
</script></body></html>"""


def _round(o, nd=4):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    return o


def build_replay_html(episodes: list[dict], path: str):
    """episodes: records from ``render.record_explained_episode`` with an extra ``label`` key."""
    slim = []
    for ep in episodes:
        e = {k: v for k, v in ep.items() if k != "steps"}
        e["steps"] = [{k: s[k] for k in ("t", "pose", "rays", "probs", "action", "ig", "shapley", "attention", "text")}
                      for s in ep["steps"]]
        slim.append(_round(e))
    html = TEMPLATE.replace("__DATA__", json.dumps(slim, separators=(",", ":")))
    with open(path, "w") as f:
        f.write(html)
    return path
