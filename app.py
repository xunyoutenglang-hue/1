# ============================================================
# 🎾 TENNIS MOTION ANALYZER PRO - ONE CELL
# 同期再生 / 動画重ね合わせ / 骨格 / 関節角度グラフ
# iPad / Safari 対応：ミュート・インライン再生・再生操作を同期
# ============================================================
import os, sys, cv2, json, base64, subprocess, urllib.request
import numpy as np
from IPython.display import display, HTML
from google.colab import files

# 1. MediaPipe準備
try:
    import mediapipe as mp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mediapipe"])
    import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = "/content/pose_landmarker_full.task"
if not os.path.exists(MODEL):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/1/pose_landmarker_full.task", MODEL
    )

# 2. 動画アップロード
print("🎾 比較したいテニス動画を2本アップロードしてください")
uploaded = files.upload()
if len(uploaded) != 2:
    raise ValueError("動画を2本アップロードしてください")

names = list(uploaded.keys())
paths = []

for i, name in enumerate(names):
    src = f"/content/source_{i}.bin"
    with open(src, "wb") as f:
        f.write(uploaded[name])
    out = f"/content/tennis_{i}.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264",
         "-pix_fmt", "yuv420p", "-acodec", "aac",
         "-movflags", "+faststart", out],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors="ignore")[-1500:])
    paths.append(out)

def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = count / fps if fps > 0 else 0
    cap.release()
    return float(duration)

dur1, dur2 = video_info(paths[0]), video_info(paths[1])

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

v1, v2 = b64(paths[0]), b64(paths[1])

# 3. 動画プレビュー
display(HTML(f"""
<div style="font-family:Arial;color:#e5e7eb;background:#111827;padding:16px;border-radius:12px">
<h3>🎯 インパクト時刻を確認</h3>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
<div><p>動画1</p><video controls playsinline style="width:100%" src="data:video/mp4;base64,{v1}"></video></div>
<div><p>動画2</p><video controls playsinline style="width:100%" src="data:video/mp4;base64,{v2}"></video></div>
</div>
<p>プレビューでボールを打つ瞬間の時刻を確認して、下の入力欄に秒数を入力してください。</p>
</div>
"""))

impact1 = float(input(f"動画1のインパクト時刻（0〜{dur1:.2f}秒）: "))
impact2 = float(input(f"動画2のインパクト時刻（0〜{dur2:.2f}秒）: "))

if not (0 <= impact1 <= dur1 and 0 <= impact2 <= dur2):
    raise ValueError("インパクト時刻が動画の長さの範囲外です")

# 4. 角度計算
def angle(a, b, c):
    a, b, c = [np.array(p, float) for p in (a, b, c)]
    u, v = a-b, c-b
    d = np.linalg.norm(u) * np.linalg.norm(v)
    if d == 0:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v)/d, -1, 1))))

# 5. 骨格解析
def analyze(path, impact, duration):
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35
    )
    detector = vision.PoseLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(path)
    ts = np.arange(max(0, impact-.75), min(duration, impact+.75)+.0001, .05)
    frames, last_ms = [], -1

    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t*1000))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(t*1000), last_ms+1)
        last_ms = ms
        result = detector.detect_for_video(img, ms)
        lm, angles = None, {k: None for k in (
            "left_elbow", "right_elbow", "left_knee",
            "right_knee", "left_hip", "right_hip"
        )}

        if result.pose_landmarks:
            p = result.pose_landmarks[0]
            lm = [{"x":float(x.x), "y":float(x.y), "z":float(x.z),
                   "visibility":float(getattr(x, "visibility", 1))}
                  for x in p]
            pt = lambda i: [p[i].x, p[i].y, p[i].z]
            angles = {
                "left_elbow": angle(pt(11),pt(13),pt(15)),
                "right_elbow": angle(pt(12),pt(14),pt(16)),
                "left_knee": angle(pt(23),pt(25),pt(27)),
                "right_knee": angle(pt(24),pt(26),pt(28)),
                "left_hip": angle(pt(11),pt(23),pt(25)),
                "right_hip": angle(pt(12),pt(24),pt(26))
            }

        frames.append({"time":float(t), "relative_time":float(t-impact),
                       "landmarks":lm, "angles":angles})

    cap.release()
    detector.close()
    return frames

print("動画1を解析中...")
data1 = analyze(paths[0], impact1, dur1)
print("動画2を解析中...")
data2 = analyze(paths[1], impact2, dur2)
print("解析完了。アプリを表示します。")

# 6. Webアプリ
html = r"""
<div id="tennis-app">
<style>
#tennis-app{font-family:Arial,"Noto Sans JP",sans-serif;background:#0b1220;color:#e5e7eb;border-radius:16px;padding:18px;overflow:hidden}
#tennis-app *{box-sizing:border-box}
.ta-header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;border-bottom:1px solid #253148;padding-bottom:14px}
.ta-title{color:#5eead4;font-size:22px;font-weight:800}
.ta-sub,.ta-label{color:#94a3b8;font-size:12px;margin-top:5px}
.ta-layout{display:grid;grid-template-columns:minmax(0,1fr);gap:14px;margin-top:16px}
.ta-card{background:#111c2e;border:1px solid #26354c;border-radius:12px;padding:14px;min-width:0}
.ta-video-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:10px}
.ta-video-wrap{position:relative;width:100%;aspect-ratio:16/9;background:#000;border-radius:9px;overflow:hidden}
.ta-video-wrap video,.ta-video-wrap canvas{position:absolute;inset:0;width:100%;height:100%}
.ta-video-wrap video{object-fit:contain;background:#000}
#ta-video1{z-index:1;opacity:1}
#ta-video2{z-index:2;opacity:.42}
#ta-canvas1{z-index:3;pointer-events:none}
#ta-canvas2{z-index:4;pointer-events:none}
.ta-controls{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:8px;margin-top:14px}
.ta-btn{border:0;border-radius:8px;padding:10px 14px;background:#203047;color:#e5e7eb;cursor:pointer;font-weight:700}
.ta-btn.primary{background:#14b8a6;color:#062b2b}
.ta-range{width:100%;accent-color:#2dd4bf}
.ta-slider-row{display:grid;grid-template-columns:65px minmax(0,1fr) 65px;gap:8px;align-items:center;margin-top:10px}
.ta-time{color:#94a3b8;font-size:12px;text-align:center}
.ta-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px}
.ta-metric{padding:12px;background:#0b1424;border-radius:9px;border:1px solid #26354c;min-width:0}
.ta-metric-name{color:#94a3b8;font-size:11px}
.ta-metric-value{color:#5eead4;font-size:22px;font-weight:800;margin-top:5px}
.ta-metric-unit{color:#94a3b8;font-size:11px}
.ta-graphs{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.ta-graph-box{min-width:0}
.ta-graph-title{font-size:13px;font-weight:700;margin-bottom:6px}
.ta-graph{width:100%;height:170px;display:block}
.ta-legend{display:flex;gap:12px;flex-wrap:wrap;color:#94a3b8;font-size:11px;margin-top:5px}
.ta-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.ta-status{color:#94a3b8;font-size:12px;text-align:center;margin-top:9px}
@media(max-width:600px){#tennis-app{padding:10px}.ta-graphs{grid-template-columns:1fr}.ta-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.ta-title{font-size:18px}}
</style>
<div class="ta-header">
 <div><div class="ta-title">🎾 TENNIS MOTION ANALYZER PRO</div><div class="ta-sub">インパクト基準・骨格比較・関節角度解析</div></div>
 <div class="ta-sub" id="ta-status">読み込み中...</div>
</div>
<div class="ta-layout">
<section class="ta-card">
<h3>🎬 同期動画比較</h3>
<div class="ta-video-grid">
 <div><div class="ta-video-name">VIDEO 1 ＋ VIDEO 2（重ね合わせ）</div>
  <div class="ta-video-wrap" id="ta-overlay-wrap">
   <video id="ta-video1" playsinline webkit-playsinline preload="auto" muted></video>
   <video id="ta-video2" playsinline webkit-playsinline preload="auto" muted></video>
   <canvas id="ta-canvas1"></canvas><canvas id="ta-canvas2"></canvas>
  </div>
 </div>
</div>
<div class="ta-controls">
 <button class="ta-btn" id="ta-back">−0.1秒</button>
 <button class="ta-btn primary" id="ta-play">▶ 再生</button>
 <button class="ta-btn" id="ta-forward">＋0.1秒</button>
 <button class="ta-btn" id="ta-impact">🎯 インパクトへ</button>
 <button class="ta-btn" id="ta-reset">⏮ 最初へ</button>
</div>
<div class="ta-slider-row"><span class="ta-time" id="ta-current">-0.75秒</span>
<input class="ta-range" id="ta-timeline" type="range" min="-0.75" max="0.75" step="0.01" value="-0.75">
<span class="ta-time">+0.75秒</span></div>
<div class="ta-status" id="ta-sync-status">準備中...</div>
</section>
<section class="ta-card">
<h3>🦴 骨格の重ね合わせ</h3>
<div class="ta-label">動画2の透明度</div><input class="ta-range" id="ta-opacity" type="range" min="0" max="1" step="0.05" value=".42">
<div class="ta-controls" style="justify-content:flex-start">
<button class="ta-btn primary" id="ta-skeleton">骨格 ON</button>
<button class="ta-btn primary" id="ta-overlay">動画重ね合わせ ON</button>
</div>
<div class="ta-metrics">
<div class="ta-metric"><div class="ta-metric-name">動画1 時刻</div><div class="ta-metric-value" id="ta-time1">0.00</div><div class="ta-metric-unit">秒</div></div>
<div class="ta-metric"><div class="ta-metric-name">動画2 時刻</div><div class="ta-metric-value" id="ta-time2">0.00</div><div class="ta-metric-unit">秒</div></div>
<div class="ta-metric"><div class="ta-metric-name">相対時間</div><div class="ta-metric-value" id="ta-relative">-0.75</div><div class="ta-metric-unit">インパクト基準 秒</div></div>
</div>
</section>
<section class="ta-card"><h3>📐 現在の関節角度</h3><div class="ta-metrics" id="ta-angle-metrics"></div></section>
<section class="ta-card"><h3>📊 関節角度の時間変化</h3><div class="ta-graphs" id="ta-graphs"></div>
<div class="ta-legend"><span><span class="ta-dot" style="background:#2dd4bf"></span>動画1</span><span><span class="ta-dot" style="background:#fb923c"></span>動画2</span><span>横軸：インパクトからの相対時間</span></div>
</section>
</div>
</div>
<script>
(function(){
const D1=__DATA1__,D2=__DATA2__,I1=__I1__,I2=__I2__,L1=__L1__,L2=__L2__;
const v1=document.getElementById("ta-video1"),v2=document.getElementById("ta-video2");
const c1=document.getElementById("ta-canvas1"),c2=document.getElementById("ta-canvas2");
const x1=c1.getContext("2d"),x2=c2.getContext("2d");
const play=document.getElementById("ta-play"),slider=document.getElementById("ta-timeline");
v1.src="data:video/mp4;base64,__V1__";v2.src="data:video/mp4;base64,__V2__";
// iOS / iPad Safari向け設定：音声なし・インライン再生
[v1,v2].forEach(v=>{v.muted=true;v.defaultMuted=true;v.playsInline=true;v.setAttribute("muted","");v.setAttribute("playsinline","");v.setAttribute("webkit-playsinline","");});
let paused=true,syncing=false,skel=true,overlay=true,raf=null,lastSync=0;
const names={left_elbow:"左ひじ",right_elbow:"右ひじ",left_knee:"左ひざ",right_knee:"右ひざ",left_hip:"左股関節",right_hip:"右股関節"};
const links=[[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[29,31],[27,31],[24,26],[26,28],[28,30],[30,32],[28,32]];
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const rel=()=>v1.currentTime-I1;
function ready(v){return new Promise((resolve,reject)=>{if(v.readyState>=2&&Number.isFinite(v.duration))return resolve();let done=false;const clean=()=>{v.removeEventListener("canplay",ok);v.removeEventListener("error",bad)};const ok=()=>{if(!done&&v.readyState>=2){done=true;clean();resolve()}};const bad=()=>{if(!done){done=true;clean();reject(Error("動画読み込みエラー"))}};v.addEventListener("canplay",ok);v.addEventListener("error",bad);v.load();setTimeout(()=>{ok();if(!done)bad()},30000)})}
function nearest(data,r){let best=null,dist=Infinity;for(const f of data){const d=Math.abs(f.relative_time-r);if(d<dist){best=f;dist=d}}return best}
function draw(canvas,ctx,frame,color,video){
 const box=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(1,Math.round(box.width*dpr)),h=Math.max(1,Math.round(box.height*dpr));
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
 ctx.clearRect(0,0,w,h);if(!skel||!frame||!frame.landmarks)return;
 const vw=video.videoWidth||w,vh=video.videoHeight||h;
 const scale=Math.min(w/vw,h/vh),rw=vw*scale,rh=vh*scale,ox=(w-rw)/2,oy=(h-rh)/2;
 const lm=frame.landmarks;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=Math.max(2,w*.0035);ctx.lineCap="round";
 for(const [a,b] of links){if(!lm[a]||!lm[b]||lm[a].visibility<.15||lm[b].visibility<.15)continue;ctx.beginPath();ctx.moveTo(ox+lm[a].x*rw,oy+lm[a].y*rh);ctx.lineTo(ox+lm[b].x*rw,oy+lm[b].y*rh);ctx.stroke()}
 for(const p of lm){if(!p||p.visibility<.15)continue;ctx.beginPath();ctx.arc(ox+p.x*rw,oy+p.y*rh,Math.max(2,w*.0045),0,Math.PI*2);ctx.fill()}
}
function angles(f1,f2){document.getElementById("ta-angle-metrics").innerHTML=Object.keys(names).map(k=>{let a=f1?.angles[k],b=f2?.angles[k];return `<div class="ta-metric"><div class="ta-metric-name">${names[k]}</div><div class="ta-metric-value">${a==null?"--":a.toFixed(0)+"°"}</div><div class="ta-metric-unit">動画2：${b==null?"--":b.toFixed(0)+"°"}</div></div>`}).join("")}
function graph(canvas,key){
 const ctx=canvas.getContext("2d"),dpr=devicePixelRatio||1,box=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.round(box.width*dpr));canvas.height=Math.max(1,Math.round(box.height*dpr));
 const w=canvas.width,h=canvas.height,l=32*dpr,r=8*dpr,t=12*dpr,b=25*dpr,pw=w-l-r,ph=h-t-b,X=x=>l+(x+.75)/1.5*pw,Y=y=>t+(1-y/180)*ph;
 ctx.clearRect(0,0,w,h);ctx.font=`${10*dpr}px Arial`;ctx.fillStyle="#94a3b8";ctx.strokeStyle="#26354c";
 [0,45,90,135,180].forEach(y=>{ctx.beginPath();ctx.moveTo(l,Y(y));ctx.lineTo(w-r,Y(y));ctx.stroke();ctx.fillText(y,2*dpr,Y(y)+3*dpr)});
 [-.75,-.5,-.25,0,.25,.5,.75].forEach(x=>{ctx.beginPath();ctx.moveTo(X(x),t);ctx.lineTo(X(x),h-b);ctx.stroke();ctx.fillText(x.toFixed(2),X(x)-12*dpr,h-7*dpr)});
 function line(data,color){ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2*dpr;let started=false;for(const f of data){const y=f.angles[key];if(y==null){started=false;continue}if(!started){ctx.moveTo(X(f.relative_time),Y(y));started=true}else ctx.lineTo(X(f.relative_time),Y(y))}ctx.stroke()}
 line(D1,"#2dd4bf");line(D2,"#fb923c");ctx.strokeStyle="#fff";ctx.setLineDash([4*dpr,4*dpr]);ctx.beginPath();ctx.moveTo(X(rel()),t);ctx.lineTo(X(rel()),h-b);ctx.stroke();ctx.setLineDash([])
}
function graphs(){const box=document.getElementById("ta-graphs");box.innerHTML="";Object.keys(names).forEach(k=>{const d=document.createElement("div");d.className="ta-graph-box";const title=document.createElement("div");title.className="ta-graph-title";title.textContent=names[k];const c=document.createElement("canvas");c.className="ta-graph";d.append(title,c);box.appendChild(d);graph(c,k)})}
function refresh(){const r=rel(),f1=nearest(D1,r),f2=nearest(D2,r);draw(c1,x1,f1,"#2dd4bf",v1);draw(c2,x2,f2,"#fb923c",v2);angles(f1,f2);document.getElementById("ta-current").textContent=(r>=0?"+":"")+r.toFixed(2)+"秒";document.getElementById("ta-time1").textContent=v1.currentTime.toFixed(2);document.getElementById("ta-time2").textContent=v2.currentTime.toFixed(2);document.getElementById("ta-relative").textContent=(r>=0?"+":"")+r.toFixed(2);if(!slider.matches(":active"))slider.value=clamp(r,-.75,.75);document.querySelectorAll(".ta-graph").forEach((c,i)=>graph(c,Object.keys(names)[i]))}
function seek(r){r=clamp(r,-I1,L1-I1);syncing=true;v1.currentTime=clamp(I1+r,0,L1);v2.currentTime=clamp(I2+r,0,L2);syncing=false;refresh()}
function pauseBoth(){paused=true;v1.pause();v2.pause();play.textContent="▶ 再生";document.getElementById("ta-sync-status").textContent="一時停止中";if(raf){cancelAnimationFrame(raf);raf=null}}
function syncLoop(){if(paused)return;if(v1.ended||v2.ended){pauseBoth();return}const target=clamp(I2+rel(),0,L2);if(Math.abs(v2.currentTime-target)>.08){syncing=true;v2.currentTime=target;syncing=false}refresh();raf=requestAnimationFrame(syncLoop)}
function startPlayback(){
 if(paused)return;
 // currentTimeの変更後、同じクリック処理内で両方のplay()を呼び出す
 const target=clamp(I2+rel(),0,L2);
 try{if(Math.abs(v2.currentTime-target)>.03)v2.currentTime=target;}catch(e){console.warn("動画2の位置合わせ:",e)}
 let p1,p2;
 try{
   p1=v1.play();
   p2=v2.play();
 }catch(e){
   pauseBoth();document.getElementById("ta-sync-status").textContent="再生できません。もう一度再生ボタンを押してください";console.error(e);return;
 }
 play.textContent="⏸ 一時停止";
 document.getElementById("ta-sync-status").textContent="2本の動画を同期再生中";
 Promise.all([Promise.resolve(p1),Promise.resolve(p2)]).then(()=>{
   if(paused)return;
   if(raf)cancelAnimationFrame(raf);
   syncLoop();
 }).catch(e=>{
   console.error("再生エラー:",e);
   pauseBoth();
   document.getElementById("ta-sync-status").textContent="再生エラー：再生ボタンをもう一度押してください";
 });
}
function toggle(){
 if(!v1.paused||!v2.paused){pauseBoth();return;}
 if(v1.readyState<2||v2.readyState<2){
   document.getElementById("ta-sync-status").textContent="動画を読み込み中です。準備完了後にもう一度押してください";
   return;
 }
 paused=false;
 startPlayback();
}
play.addEventListener("click",toggle);
document.getElementById("ta-back").onclick=()=>{pauseBoth();seek(rel()-.1)};
document.getElementById("ta-forward").onclick=()=>{pauseBoth();seek(rel()+.1)};
document.getElementById("ta-impact").onclick=()=>{pauseBoth();seek(0)};
document.getElementById("ta-reset").onclick=()=>{pauseBoth();seek(-.75)};
slider.addEventListener("input",()=>{pauseBoth();seek(Number(slider.value))});
document.getElementById("ta-opacity").addEventListener("input",e=>v2.style.opacity=e.target.value);
document.getElementById("ta-skeleton").onclick=e=>{skel=!skel;e.target.textContent=skel?"骨格 ON":"骨格 OFF";e.target.classList.toggle("primary",skel);refresh()};
document.getElementById("ta-overlay").onclick=e=>{overlay=!overlay;v2.style.display=overlay?"block":"none";e.target.textContent=overlay?"動画重ね合わせ ON":"動画重ね合わせ OFF";e.target.classList.toggle("primary",overlay)};
v1.addEventListener("timeupdate",()=>{if(!paused&&!syncing&&performance.now()-lastSync>80){lastSync=performance.now();const t=clamp(I2+rel(),0,L2);if(Math.abs(v2.currentTime-t)>.08){syncing=true;v2.currentTime=t;syncing=false}}refresh()});
v1.addEventListener("ended",pauseBoth);v2.addEventListener("ended",pauseBoth);
window.addEventListener("resize",refresh);
Promise.all([ready(v1),ready(v2)]).then(()=>{
 v1.currentTime=clamp(I1-.75,0,L1);v2.currentTime=clamp(I2-.75,0,L2);
 graphs();refresh();
 document.getElementById("ta-status").textContent="準備完了";
 document.getElementById("ta-sync-status").textContent="準備完了：再生ボタンを押してください";
}).catch(e=>{
 document.getElementById("ta-status").textContent="読み込みエラー";
 document.getElementById("ta-sync-status").textContent="動画の読み込みに失敗しました";
 console.error(e);
});
})();
</script>
"""

html = (html.replace("__DATA1__", json.dumps(data1, separators=(",", ":")))
    .replace("__DATA2__", json.dumps(data2, separators=(",", ":")))
    .replace("__I1__", str(impact1)).replace("__I2__", str(impact2))
    .replace("__L1__", str(dur1)).replace("__L2__", str(dur2))
    .replace("__V1__", v1).replace("__V2__", v2))

display(HTML(html))
