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
        :root {
            --bg: #f0f2f5;
            --card: #ffffff;
            --text: #1a2b3c;
            --muted: #64748b;
            --accent: #6366f1;
            --danger: #ef4444;
            --ok: #22c55e;
            --glass: rgba(0,0,0,0.04);
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { 
            height: 100%;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
        }
        .wrap {
            min-height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }
        .card {
            width: 100%;
            max-width: 900px;
            background: var(--card);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: var(--shadow);
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .title {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        .logo {
            width: 56px;
            height: 56px;
            border-radius: 16px;
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 800;
            font-size: 1.25rem;
            box-shadow: 0 10px 20px -5px rgba(79, 70, 229, 0.3);
        }
        h1 {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text);
            margin: 0;
        }
        .sub {
            color: var(--muted);
            font-size: 0.875rem;
        }
        .input-group {
            position: relative;
            display: flex;
            gap: 1rem;
            margin: 1.5rem 0;
        }
        input[type="text"] {
            flex: 1;
            padding: 0.875rem 1rem;
            font-size: 1rem;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            background: white;
            color: var(--text);
            transition: all 0.2s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        .btn {
            padding: 0.875rem 1.5rem;
            font-size: 0.875rem;
            font-weight: 600;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        .btn-primary {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
            box-shadow: 0 4px 12px -2px rgba(79, 70, 229, 0.3);
        }
        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 15px -3px rgba(79, 70, 229, 0.4);
        }
        .btn-secondary {
            background: #f8fafc;
            color: var(--text);
            border: 2px solid #e2e8f0;
        }
        .btn-secondary:hover {
            background: #f1f5f9;
        }
        .result {
            margin-top: 2rem;
            padding: 2rem;
            border-radius: 16px;
            transition: all 0.3s;
        }
        .result.ham {
            background: rgba(34, 197, 94, 0.05);
            border: 2px solid rgba(34, 197, 94, 0.1);
        }
        .result.spam {
            background: rgba(239, 68, 68, 0.05);
            border: 2px solid rgba(239, 68, 68, 0.1);
        }
        .result-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .badge {
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .badge.ham {
            background: rgba(34, 197, 94, 0.1);
            color: #15803d;
        }
        .badge.spam {
            background: rgba(239, 68, 68, 0.1);
            color: #b91c1c;
        }
        .alert {
            padding: 1rem;
            border-radius: 12px;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            animation: slideIn 0.3s ease-out;
        }
        .alert.ham {
            background: #f0fdf4;
            border: 2px solid #dcfce7;
            color: #15803d;
        }
        .alert.spam {
            background: #fef2f2;
            border: 2px solid #fee2e2;
            color: #b91c1c;
        }
        .alert-icon {
            width: 24px;
            height: 24px;
            flex-shrink: 0;
        }
        .prob-bars {
            margin-top: 1.5rem;
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 12px;
        }
        .prob-row {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-top: 0.75rem;
        }
        .prob-label {
            width: 80px;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text);
        }
        .prob-bar {
            flex: 1;
            height: 8px;
            background: #e2e8f0;
            border-radius: 9999px;
            overflow: hidden;
        }
        .prob-bar > div {
            height: 100%;
            transition: width 0.3s ease-out;
        }
        .prob-bar.spam > div { background: linear-gradient(90deg, #ef4444, #f43f5e); }
        .prob-bar.ham > div { background: linear-gradient(90deg, #22c55e, #10b981); }
        .prob-value {
            min-width: 60px;
            font-size: 0.875rem;
            font-weight: 600;
            text-align: right;
        }
        .status {
            padding: 0.75rem 1rem;
            background: #f8fafc;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 0.875rem;
            color: var(--text);
        }
        .status strong {
            color: var(--accent);
        }
        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid #e2e8f0;
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg) } }
        @keyframes slideIn { from { opacity: 0; transform: translateY(-10px) } to { opacity: 1; transform: translateY(0) } }
        @media (max-width: 640px) {
            .wrap { padding: 1rem; }
            .card { padding: 1.5rem; }
            .header { flex-direction: column; align-items: flex-start; }
            .input-group { flex-direction: column; }
            .btn { width: 100%; justify-content: center; }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="card">
            <div class="header">
                <div class="title">
                    <div class="logo">AI</div>
                    <div>
                        <h1>SMS Spam Detector</h1>
                        <div class="sub">Powered by Machine Learning · Instant Detection</div>
                    </div>
                </div>
                <div class="status" id="statusBox">
                    Model Status: <strong id="modelState">checking...</strong>
                </div>
            </div>

            <div class="input-group">
                <input id="msg" type="text" 
                    placeholder="Type or paste a message to check (e.g., 'You won 1 lakh, claim now')" 
                    autocomplete="off" />
                <button id="predBtn" class="btn btn-primary">
                    Analyze Message
                </button>
            </div>

            <div id="resultBox" style="display:none" class="result">
                <div class="result-header">
                    <div id="label" class="badge">--</div>
                    <div id="probText" style="color: var(--muted)"></div>
                    <div id="busy" style="display:none">
                        <span class="spinner"></span>
                    </div>
                </div>
                
                <div id="alertBox" class="alert" style="display:none">
                    <svg class="alert-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                    </svg>
                    <div id="alertText"></div>
                </div>

                <div class="prob-bars">
                    <div style="font-weight:500;margin-bottom:1rem">Confidence Scores</div>
                    <div id="probsBox"></div>
                </div>
            </div>

            <div style="display:flex;gap:1rem;margin-top:2rem;justify-content:flex-end">
                <button id="resetBtn" class="btn btn-secondary">Reset Model</button>
                <button id="trainBtn" class="btn btn-primary">Train Model</button>
            </div>

            <div style="margin-top:2rem;padding-top:2rem;border-top:2px solid #f1f5f9">
                <div style="font-weight:500;margin-bottom:0.75rem">How it works</div>
                <div style="color:var(--muted);font-size:0.875rem">
                    This demo uses a Naive Bayes classifier trained on SMS messages.
                    Click "Train Model" to initialize with sample data, then enter any message to analyze.
                    The model will classify it as either legitimate (HAM) or suspicious (SPAM).
                </div>
            </div>
        </div>
    </div>

        </div>
    </div>
    <script>
        const el = id => document.getElementById(id);
        
        async function check() {
            try {
                const r = await fetch('/api/status');
                const j = await r.json();
                el('modelState').textContent = j.trained ? 'Ready' : 'Not Trained';
                el('modelState').style.color = j.trained ? '#22c55e' : '#ef4444';
            } catch(e) {
                el('modelState').textContent = 'Error';
                el('modelState').style.color = '#ef4444';
            }
        }

        function showBusy(on) {
            el('busy').style.display = on ? 'inline-block' : 'none';
            el('predBtn').disabled = on;
            el('trainBtn').disabled = on;
        }

        function showResult(show) {
            el('resultBox').style.display = show ? 'block' : 'none';
            if (!show) {
                el('alertBox').style.display = 'none';
            }
        }

        function renderProbs(probs) {
            const box = el('probsBox');
            box.innerHTML = '';
            const entries = Object.entries(probs||{}).sort((a,b) => b[1]-a[1]);
            
            for (const [k,v] of entries) {
                const row = document.createElement('div');
                row.className = 'prob-row';
                
                const label = document.createElement('div');
                label.className = 'prob-label';
                label.textContent = k.toUpperCase();
                
                const barWrap = document.createElement('div');
                barWrap.className = `prob-bar ${k.toLowerCase()}`;
                
                const inner = document.createElement('div');
                inner.style.width = Math.max(2, Math.round(v*100)) + '%';
                barWrap.appendChild(inner);
                
                const value = document.createElement('div');
                value.className = 'prob-value';
                value.textContent = (v*100).toFixed(1) + '%';
                
                row.appendChild(label);
                row.appendChild(barWrap);
                row.appendChild(value);
                box.appendChild(row);
            }
        }

        async function train() {
            showBusy(true);
            showResult(false);
            try {
                const r = await fetch('/train', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: 'n=20'
                });
                const text = await r.text();
                el('alertBox').className = 'alert ham';
                el('alertText').textContent = text;
                el('alertBox').style.display = 'flex';
            } catch(e) {
                el('alertBox').className = 'alert spam';
                el('alertText').textContent = 'Training failed. Please try again.';
                el('alertBox').style.display = 'flex';
            }
            showBusy(false);
            check();
        }

        async function predict() {
            const msg = el('msg').value.trim();
            if (!msg) {
                el('alertBox').className = 'alert spam';
                el('alertText').textContent = 'Please enter a message to analyze';
                el('alertBox').style.display = 'flex';
                showResult(true);
                return;
            }

            showBusy(true);
            try {
                const r = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: 'message=' + encodeURIComponent(msg)
                });
                const j = await r.json();
                
                if (j.error) {
                    el('alertBox').className = 'alert spam';
                    el('alertText').textContent = j.error;
                    el('alertBox').style.display = 'flex';
                    showResult(true);
                    return;
                }

                const isSpam = j.predicted === 'spam';
                el('resultBox').className = 'result ' + (isSpam ? 'spam' : 'ham');
                el('label').textContent = j.predicted.toUpperCase();
                el('label').className = 'badge ' + (isSpam ? 'spam' : 'ham');
                
                const probs = Object.entries(j.probs||{});
                el('probText').textContent = probs.map(([k,v]) => 
                    `${k.toUpperCase()}: ${(v*100).toFixed(1)}%`
                ).join(' | ');
                
                el('alertBox').className = 'alert ' + (isSpam ? 'spam' : 'ham');
                el('alertText').textContent = isSpam ? 
                    'Warning: This message appears to be spam. Be cautious!' :
                    'This message appears to be legitimate.';
                el('alertBox').style.display = 'flex';
                
                renderProbs(j.probs||{});
                showResult(true);
            } catch(e) {
                el('alertBox').className = 'alert spam';
                el('alertText').textContent = 'Analysis failed. Please try again.';
                el('alertBox').style.display = 'flex';
                showResult(true);
            }
            showBusy(false);
        }

        function clearModel() {
            if (!confirm('Are you sure you want to reset the model? You will need to retrain.')) return;
            
            fetch('/clear', {method:'POST'}).then(() => {
                showResult(false);
                el('msg').value = '';
                check();
                el('alertBox').className = 'alert ham';
                el('alertText').textContent = 'Model cleared successfully';
                el('alertBox').style.display = 'flex';
                showResult(true);
            });
        }

        // Event listeners
        el('trainBtn').addEventListener('click', train);
        el('predBtn').addEventListener('click', predict);
        el('resetBtn').addEventListener('click', clearModel);
        el('msg').addEventListener('keypress', e => {
            if (e.key === 'Enter') predict();
        });

        // Initialize
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
            probs_arr = clf.predict_proba(x)[0]
            # clf.predict_proba may return a dict (class->prob) or a sequence;
            # normalize into a mapping of class->float for JSON serialization.
            if isinstance(probs_arr, dict):
                probs = {str(k): float(v) for k, v in probs_arr.items()}
            else:
                classes = getattr(clf, 'classes_', None)
                if classes is not None:
                    probs = {str(c): float(p) for c, p in zip(classes, probs_arr)}
                else:
                    probs = [float(p) for p in probs_arr]
            out = {'message': message, 'predicted': str(pred), 'probs': probs}
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
