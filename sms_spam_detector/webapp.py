import os
import sys
import json
import csv
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from sms_spam_detector.model import SimpleCountVectorizer, MultinomialNB, save_pipeline, load_pipeline

DATA_PATH = os.path.join(ROOT, 'data', 'sms_sample_20.csv')
MODEL_PATH = os.path.join(ROOT, 'model.pkl')

HTML_PAGE = '''<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>SMS Spam Detector</title>
    <style>
        :root{--bg:#0b1220;--card:#0f1724;--muted:#9aa8bf;--accent:#7c3aed;--danger:#ef4444;--ok:#10b981;--glass:rgba(255,255,255,0.04)}
        *{box-sizing:border-box}
        html,body{height:100%;margin:0;font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:linear-gradient(180deg,#041025 0%, #081127 100%);color:#e6eef8}
        .wrap{min-height:100%;display:flex;align-items:center;justify-content:center;padding:40px}
        .card{width:100%;max-width:900px;border-radius:16px;padding:28px;background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));box-shadow:0 10px 40px rgba(2,6,23,0.6);border:1px solid rgba(255,255,255,0.03)}
        .header{display:flex;align-items:center;justify-content:space-between;gap:16px}
        .title{display:flex;gap:14px;align-items:center}
        .logo{width:52px;height:52px;border-radius:12px;background:linear-gradient(135deg,var(--accent),#4f46e5);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:18px}
        h1{margin:0;font-size:20px}
        .sub{color:var(--muted);font-size:13px}
        .controls{display:flex;gap:12px;margin-top:18px}
        .input{display:flex;gap:12px;align-items:center}
        input[type=text]{flex:1;padding:12px 14px;border-radius:12px;border:1px solid var(--glass);background:transparent;color:inherit;font-size:15px}
        .btn{background:linear-gradient(90deg,var(--accent),#5b21b6);border:none;color:white;padding:10px 16px;border-radius:12px;cursor:pointer;font-weight:700}
        .btn.secondary{background:transparent;border:1px solid rgba(255,255,255,0.06);color:var(--muted);font-weight:600}
        .panel{display:flex;gap:20px;margin-top:20px}
        .left{flex:1}
        .right{width:280px}
        .status{padding:12px;border-radius:10px;background:rgba(255,255,255,0.01);border:1px solid rgba(255,255,255,0.02);color:var(--muted)}
        .badge{display:inline-block;padding:12px 18px;border-radius:999px;font-weight:800;font-size:18px}
        .badge.spam{background:linear-gradient(90deg, rgba(239,68,68,0.12), rgba(239,68,68,0.06));color:var(--danger)}
        .badge.ham{background:linear-gradient(90deg, rgba(16,185,129,0.12), rgba(16,185,129,0.06));color:var(--ok)}
        .probs{margin-top:12px}
        .prob-row{display:flex;align-items:center;justify-content:space-between;margin-top:8px}
        .bar{height:10px;background:rgba(255,255,255,0.04);border-radius:999px;overflow:hidden;margin-left:10px;flex:1}
        .bar > i{display:block;height:100%;background:linear-gradient(90deg,#4f46e5,#7c3aed)}
        .small{font-size:13px;color:var(--muted)}
        .spinner{width:18px;height:18px;border-radius:50%;border:3px solid rgba(255,255,255,0.08);border-top-color:var(--accent);animation:spin 1s linear infinite;display:inline-block}
        @keyframes spin{to{transform:rotate(360deg)}}
        .copy{background:transparent;border:1px solid rgba(255,255,255,0.04);padding:8px;border-radius:8px;color:var(--muted);cursor:pointer}
        @media (max-width:780px){.panel{flex-direction:column}.right{width:100%}}
    </style>
</head>
<body>
    <div class="wrap">
        <div class="card">
            <div class="header">
                <div class="title">
                    <div class="logo">SB</div>
                    <div>
                        <h1>SMS Spam Detector</h1>
                        <div class="sub">Naive Bayes · small demo · train on 20 samples</div>
                    </div>
                </div>
                <div class="status" id="statusBox">Model: <strong id="modelState">checking...</strong></div>
            </div>

            <div class="controls">
                <div class="input left">
                    <input id="msg" type="text" placeholder="Type a message to classify (eg. 'You won 1 lakh, claim now')" autocomplete="off" />
                    <button id="predBtn" class="btn">Predict</button>
                    <button id="copyBtn" class="copy" title="Copy label">Copy</button>
                </div>
                <div style="display:flex;gap:8px;align-items:center">
                    <button id="trainBtn" class="btn">Train on 20 samples</button>
                    <button id="resetBtn" class="btn secondary">Clear Model</button>
                </div>
            </div>

            <div class="panel">
                <div class="left">
                    <div style="margin-top:18px;display:flex;align-items:center;gap:16px">
                        <div id="label" class="badge ham">--</div>
                        <div id="probText" class="small">&nbsp;</div>
                        <div id="busy" style="display:none"><span class="spinner"></span></div>
                    </div>
                    <div class="probs" id="probsBox"></div>
                </div>
                <div class="right">
                    <div class="status">Info
                        <div class="small" style="margin-top:10px">Use Train to create a model from the bundled 20-sample CSV. Predict uses the saved model file.</div>
                    </div>
                </div>
            </div>

        </div>
    </div>
    <script>
        function el(id){return document.getElementById(id)}
        async function check(){
            try{let r=await fetch('/api/status');let j=await r.json();el('modelState').textContent=j.trained? 'trained':'not trained';}catch(e){el('modelState').textContent='error'}
        }
        function showBusy(on){el('busy').style.display= on? 'inline-block':'none'}
        function renderProbs(probs){
            const box = el('probsBox'); box.innerHTML='';
            const entries = Object.entries(probs||{}).sort((a,b)=>b[1]-a[1]);
            for(const [k,v] of entries){
                const row = document.createElement('div'); row.className='prob-row';
                const left = document.createElement('div'); left.className='small'; left.textContent = k.toUpperCase();
                const barWrap = document.createElement('div'); barWrap.className='bar';
                const inner = document.createElement('i'); inner.style.width = Math.max(2, Math.round(v*100)) + '%';
                if(k==='spam') inner.style.background = 'linear-gradient(90deg,#ef4444,#f97316)';
                else inner.style.background = 'linear-gradient(90deg,#10b981,#34d399)';
                barWrap.appendChild(inner);
                row.appendChild(left); row.appendChild(barWrap);
                box.appendChild(row);
            }
        }
        async function train(){ showBusy(true); try{ let r=await fetch('/train',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'n=20'}); let t=await r.text(); alert(t); }catch(e){alert('Train failed')} showBusy(false); check(); }
        async function predict(){ const msg = el('msg').value.trim(); if(!msg){alert('Enter a message'); return;} showBusy(true); try{ let r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'message='+encodeURIComponent(msg)}); let j=await r.json(); el('label').textContent = j.predicted.toUpperCase(); el('label').className = 'badge ' + (j.predicted==='spam' ? 'spam':'ham'); el('probText').textContent = Object.entries(j.probs||{}).map(([k,v])=>`${k}: ${(v*100).toFixed(1)}%`).join(' | '); renderProbs(j.probs||{}); }catch(e){alert('Prediction failed')} showBusy(false); }
        function copyLabel(){ const txt = el('label').textContent; if(!txt||txt==='--') return; navigator.clipboard?.writeText(txt).then(()=>{alert('Copied: '+txt)}).catch(()=>{ /* ignore */ }); }
        function clearModel(){ if(!confirm('Clear model file?')) return; fetch('/clear',{method:'POST'}).then(()=>{alert('Model cleared'); check(); el('label').textContent='--'; el('probText').textContent=''; el('probsBox').innerHTML='';}); }
        el('trainBtn').addEventListener('click', train);
        el('predBtn').addEventListener('click', predict);
        el('copyBtn').addEventListener('click', copyLabel);
        el('resetBtn').addEventListener('click', clearModel);
        check();
    </script>
</body>
</html>'''

# helpers
def load_csv(path, n=20):
    texts = []
    labels = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= n:
                break
            labels.append(r['label'])
            texts.append(r['text'])
    return texts, labels


def train_model(n=20):
    texts, labels = load_csv(DATA_PATH, n=n)
    vec = SimpleCountVectorizer()
    X = vec.fit_transform(texts)
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, labels)
    save_pipeline(MODEL_PATH, vec, clf)
    return len(texts)


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    return load_pipeline(MODEL_PATH)

class Handler(BaseHTTPRequestHandler):
    def _send_html(self, html_body):
        body = html_body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/api/status'):
            vec, clf = load_model()
            data = {'trained': vec is not None}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
            return
        # serve the module-level HTML page
        self._send_html(HTML_PAGE)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8')
        params = parse_qs(body)
        if self.path == '/train':
            n = int(params.get('n', ['20'])[0])
            count = train_model(n=n)
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f'Trained on {count} samples'.encode('utf-8'))
            return
        if self.path == '/clear':
            # remove model file if exists
            try:
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                self.send_response(200)
                self.end_headers()
            except Exception:
                self.send_response(500)
                self.end_headers()
            return
        if self.path == '/predict':
            message = params.get('message', [''])[0]
            vec, clf = load_model()
            if vec is None:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error':'model not trained yet'}).encode('utf-8'))
                return
            x = vec.transform([message])
            pred = clf.predict(x)[0]
            probs = clf.predict_proba(x)[0]
            out = {'message': message, 'predicted': pred, 'probs': probs}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(out).encode('utf-8'))
            return
        # fallback
        self.send_response(404)
        self.end_headers()

if __name__ == '__main__':
    port = 8000
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f'Serving on http://localhost:{port} - open in your browser')
    server.serve_forever()
