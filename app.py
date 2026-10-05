import json
import os
from pathlib import Path
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from audio import AudioStore, parse_cuts_param
from offsets import OffsetStore
from security import SessionManager, request_token, resolves_publicly
from sync import SyncEngine


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(os.getenv("SIDECAR_CACHE_DIR", APP_DIR / "data"))
PUBLIC_BASE_URL = os.getenv("SIDECAR_PUBLIC_URL", "").strip().rstrip("/")
SESSION_TTL = int(os.getenv("SIDECAR_SESSION_TTL", "21600"))
FIXED_TOKEN = os.getenv("SIDECAR_FIXED_TOKEN", "").strip()
AUDIO_PROXY = os.getenv("SIDECAR_AUDIO_PROXY", "").strip()
OFFSET_API_URL = os.getenv("OFFSET_API_URL", "").strip()
OFFSET_API_TOKEN = os.getenv("OFFSET_API_TOKEN", "").strip()
BOOTSTRAP_KEY = os.getenv("SIDECAR_BOOTSTRAP_KEY", "").strip()

audio = AudioStore(str(CACHE_DIR / "audio"), proxy=AUDIO_PROXY)
offsets = OffsetStore(str(CACHE_DIR / "offsets.db"), OFFSET_API_URL, OFFSET_API_TOKEN)
sessions = SessionManager(SESSION_TTL, FIXED_TOKEN)
sync_engine = SyncEngine(audio, offsets, AUDIO_PROXY)

app = FastAPI(title="Toast Audio Sidecar", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[item.strip() for item in os.getenv("CORS_ORIGINS", "*").split(",") if item.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _require_session(request: Request, body: dict | None = None) -> str:
    token = request_token(request, body)
    if not sessions.valid(token):
        raise HTTPException(status_code=401, detail="sidecar session required")
    return token


def _base_url(request: Request) -> str:
    return PUBLIC_BASE_URL or str(request.base_url).rstrip("/")


def _audio_url(request: Request, hid: str, token: str, offset: float = 0.0, rate: float = 1.0) -> str:
    query = urlencode({"o": int(round(offset * 1000)), "r": int(round(rate * 1_000_000_000)), "t": token})
    return f"{_base_url(request)}/dual/aud/{hid}/audio.m3u8?{query}"


LANDING_HTML = """<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ToastFlix Audio Sidecar</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🍞</text></svg>">
    <style>
        :root {
            --bg: #0e0e11;
            --card-bg: #18181c;
            --border: #2e2e36;
            --text: #f4f4f6;
            --text-muted: #9494a0;
            --accent: #ff9800;
            --accent-purple: #7952ff;
            --green: #22c55e;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 24px 16px;
        }
        .container {
            max-width: 640px;
            width: 100%;
            background: var(--card-bg);
            border: 2px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .header {
            padding: 18px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255, 255, 255, 0.02);
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-icon {
            font-size: 26px;
            line-height: 1;
        }
        .brand-title {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #fff;
        }
        .brand-subtitle {
            font-size: 0.78rem;
            color: var(--text-muted);
            font-weight: 500;
        }
        .badge-status {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(34, 197, 94, 0.12);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .badge-status::before {
            content: '';
            width: 8px;
            height: 8px;
            background: #22c55e;
            border-radius: 50%;
            box-shadow: 0 0 8px #22c55e;
        }
        .media-box {
            position: relative;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            border-bottom: 1px solid var(--border);
        }
        .media-box img {
            width: 100%;
            height: auto;
            display: block;
            object-fit: cover;
            max-height: 440px;
        }
        .content {
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .info-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            font-size: 0.90rem;
            line-height: 1.5;
            color: #d1d1db;
        }
        .info-card strong {
            color: #fff;
        }
        .url-box {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #09090b;
            border: 1px solid #3f3f46;
            border-radius: 8px;
            padding: 10px 14px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.85rem;
            color: #38bdf8;
            word-break: break-all;
            margin-top: 10px;
        }
        .actions {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 0.86rem;
            font-weight: 700;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-primary {
            background: var(--accent-purple);
            color: #fff;
            border: none;
        }
        .btn-primary:hover {
            background: #653be0;
            transform: translateY(-1px);
        }
        .btn-secondary {
            background: #27272a;
            color: #f4f4f6;
            border: 1px solid #3f3f46;
        }
        .btn-secondary:hover {
            background: #3f3f46;
            transform: translateY(-1px);
        }
        .url-action-row {
            margin-top: 12px;
            display: flex;
        }
        .btn-toastflix {
            width: 100%;
            background: #ff9800;
            color: #000;
            border: 2px solid #000;
            box-shadow: 3px 3px 0 #000;
            padding: 12px 18px;
            font-size: 0.95rem;
            font-weight: 900;
            cursor: pointer;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.15s ease;
        }
        .btn-toastflix:hover {
            background: #ffa834;
            transform: translate(-1px, -1px);
            box-shadow: 4px 4px 0 #000;
        }
        .btn-toastflix:active {
            transform: translate(2px, 2px);
            box-shadow: 1px 1px 0 #000;
        }
        .btn-toastflix.copied {
            background: #22c55e;
            color: #000;
        }
        .footer {
            padding: 14px 24px;
            text-align: center;
            font-size: 0.75rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            background: rgba(0, 0, 0, 0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="brand">
                <span class="brand-icon">🍞</span>
                <div>
                    <h1 class="brand-title">ToastFlix Sidecar</h1>
                    <div class="brand-subtitle">Audio Remux & Sync Microservice</div>
                </div>
            </div>
            <div class="badge-status">Online</div>
        </div>

        <div class="media-box">
            <img src="https://i.imgur.com/nGZPk3R.jpeg" alt="ToastFlix Sidecar">
        </div>

        <div class="content">
            <div class="info-card">
                <strong>Microservizio Audio Attivo!</strong><br>
                Questo server gestisce l'estrazione, la conversione e la sincronizzazione delle tracce audio italiane per i flussi <strong>4K / FHD Remuxed Dual Audio</strong> di ToastFlix.
                
                <div class="url-box">
                    <span id="urlText">Rilevamento indirizzo...</span>
                </div>

                <div class="url-action-row">
                    <button type="button" class="btn-toastflix" id="btnCopyInsert" onclick="copyAndInsertToastflix()">
                        📋 Copia e inserisci in Toastflix
                    </button>
                </div>
            </div>

            <div class="info-card" id="connectivityCard">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong>📡 Connettività Provider Audio</strong>
                    <span id="proxyBadge" style="font-size:0.75rem; padding:2px 8px; border-radius:6px; background:#27272a; color:#a1a1aa;">WARP Fonte 2: verifica...</span>
                </div>
                <div style="margin-top:8px; font-size:0.85rem; color:#a1a1aa; line-height:1.4;">
                    Verifica lo stato delle due sorgenti audio italiane per ToastFlix:
                    <ul style="margin:6px 0 0 18px; padding:0; color:#d1d1db;">
                        <li><strong>Fonte 1</strong>: Stream HLS primario (connessione sempre diretta).</li>
                        <li><strong>Fonte 2</strong>: Stream AAC secondario (richiede WARP/proxy se l'IP è filtrato).</li>
                    </ul>
                </div>
                <div id="testResultBox" style="display:none; margin-top:12px; font-size:0.85rem; line-height:1.4;"></div>
                <div style="margin-top:12px;">
                    <button type="button" class="btn btn-secondary" id="btnTestProviders" onclick="testProvidersConnectivity()" style="width:100%;">
                        🔍 Verifica Connettività (Fonte 1 & Fonte 2)
                    </button>
                </div>
            </div>

            <div class="actions">
                <a href="https://github.com/qwertyuiop8899/toastflix-sidecar" target="_blank" rel="noopener noreferrer" class="btn btn-primary">
                    📖 Documentazione GitHub
                </a>
                <a href="/health" class="btn btn-secondary">
                    🩺 Health Check
                </a>
            </div>
        </div>

        <div class="footer">
            ToastFlix Community · Powered by FastAPI & FFmpeg
        </div>
    </div>

    <script>
        var serverForcedUrl = "__PUBLIC_BASE_URL__";

        document.addEventListener('DOMContentLoaded', function() {
            var urlSpan = document.getElementById('urlText');
            var effectiveUrl = (serverForcedUrl && serverForcedUrl.trim()) ? serverForcedUrl.trim() : window.location.origin;
            if (urlSpan) {
                urlSpan.textContent = effectiveUrl;
            }
            fetch('/api/test-providers?quick=1')
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    var badge = document.getElementById('proxyBadge');
                    if (badge) {
                        badge.textContent = d.warp_active ? 'WARP Fonte 2: Attivo' : 'WARP Fonte 2: Non attivo (Diretto)';
                        badge.style.color = d.warp_active ? '#4ade80' : '#a1a1aa';
                    }
                })
                .catch(function() {});
        });

        function copyAndInsertToastflix() {
            var effectiveUrl = (serverForcedUrl && serverForcedUrl.trim()) ? serverForcedUrl.trim() : window.location.origin;
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(effectiveUrl).catch(function() {});
            } else {
                var ta = document.createElement('textarea');
                ta.value = effectiveUrl;
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
            }
            var btn = document.getElementById('btnCopyInsert');
            if (btn) {
                btn.textContent = '✓ Copiato negli appunti!';
                btn.classList.add('copied');
                setTimeout(function() {
                    btn.textContent = '📋 Copia e inserisci in Toastflix';
                    btn.classList.remove('copied');
                }, 2500);
            }
        }

        function testProvidersConnectivity() {
            var btn = document.getElementById('btnTestProviders') || document.getElementById('btnTestPartite');
            var resBox = document.getElementById('testResultBox');
            var badge = document.getElementById('proxyBadge');
            btn.disabled = true;
            btn.textContent = '⏳ Test in corso...';
            resBox.style.display = 'none';
            fetch('/api/test-providers')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    btn.disabled = false;
                    btn.textContent = '🔍 Verifica di nuovo';
                    resBox.style.display = 'block';
                    if (badge) {
                        var hasWarp = data.proxy && data.proxy.indexOf('socks') !== -1;
                        badge.textContent = hasWarp ? 'WARP Fonte 2: Attivo' : 'WARP Fonte 2: Non attivo (Diretto)';
                        badge.style.color = hasWarp ? '#4ade80' : '#a1a1aa';
                    }
                    var f1 = data.fonte1 || {};
                    var f2 = data.fonte2 || {};
                    var f1Ok = f1.status === 'ok';
                    var f2Ok = f2.status === 'ok';

                    var html = '<div style="display:flex; flex-direction:column; gap:10px;">';

                    // Fonte 1
                    html += '<div style="padding:10px 12px; border-radius:8px; background:' + (f1Ok ? 'rgba(34, 197, 94, 0.12)' : 'rgba(239, 68, 68, 0.12)') + '; border:1px solid ' + (f1Ok ? '#22c55e' : '#ef4444') + '; color:' + (f1Ok ? '#4ade80' : '#f87171') + ';">';
                    html += '<div style="display:flex; justify-content:space-between; align-items:center;">';
                    html += '<strong>' + (f1Ok ? '✅ ' : '❌ ') + (f1.name || 'Fonte 1') + '</strong>';
                    html += '<span style="font-size:0.75rem; padding:1px 6px; border-radius:4px; background:#18181b; color:#a1a1aa;">' + (f1.mode || 'Diretto') + '</span>';
                    html += '</div>';
                    html += '<div style="margin-top:4px; font-size:0.82rem; color:' + (f1Ok ? '#bbf7d0' : '#fca5a5') + ';">' + (f1.message || '') + '</div>';
                    html += '</div>';

                    // Fonte 2
                    html += '<div style="padding:10px 12px; border-radius:8px; background:' + (f2Ok ? 'rgba(34, 197, 94, 0.12)' : 'rgba(239, 68, 68, 0.12)') + '; border:1px solid ' + (f2Ok ? '#22c55e' : '#ef4444') + '; color:' + (f2Ok ? '#4ade80' : '#f87171') + ';">';
                    html += '<div style="display:flex; justify-content:space-between; align-items:center;">';
                    html += '<strong>' + (f2Ok ? '✅ ' : '⚠️ ') + (f2.name || 'Fonte 2') + '</strong>';
                    html += '<span style="font-size:0.75rem; padding:1px 6px; border-radius:4px; background:#18181b; color:#a1a1aa;">' + (f2.proxy || 'Diretto') + '</span>';
                    html += '</div>';
                    html += '<div style="margin-top:4px; font-size:0.82rem; color:' + (f2Ok ? '#bbf7d0' : '#fca5a5') + ';">' + (f2.message || '') + '</div>';
                    if (!f2Ok && f2.suggestion) {
                        html += '<div style="margin-top:8px; padding:8px 10px; border-radius:6px; background:rgba(0,0,0,0.35); border:1px solid rgba(239,68,68,0.4); font-size:0.80rem; color:#fed7aa; line-height:1.4;">';
                        html += '💡 <strong>Suggerimento:</strong> ' + f2.suggestion;
                        html += '</div>';
                    }
                    html += '</div>';

                    html += '</div>';
                    resBox.innerHTML = html;
                })
                .catch(function(err) {
                    btn.disabled = false;
                    btn.textContent = '🔍 Riprova test';
                    resBox.style.display = 'block';
                    resBox.innerHTML = '<div style="padding:10px; border-radius:8px; background:rgba(239, 68, 68, 0.15); border:1px solid #ef4444; color:#f87171;"><strong>Errore test:</strong> ' + err + '</div>';
                });
        }
        var testPartiteConnectivity = testProvidersConnectivity;
    </script>
</body>
</html>
"""


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def index():
    forced_url = PUBLIC_BASE_URL or ""
    content = LANDING_HTML.replace("__PUBLIC_BASE_URL__", forced_url)
    return HTMLResponse(content=content)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "toast-audio-sidecar", "public_url": PUBLIC_BASE_URL or None}


@app.get("/api/test-providers")
@app.get("/api/test-partite")
async def test_providers(quick: int = 0):
    proxy = AUDIO_PROXY or os.getenv("SIDECAR_AUDIO_PROXY", "").strip()
    if quick:
        return {
            "proxy": proxy if proxy else "Diretto (nessun proxy)",
            "warp_active": bool(proxy)
        }

    # --- Fonte 1 (SEMPRE DIRETTO, mai attraverso WARP o altri proxy) ---
    fonte1 = {
        "name": "Fonte 1",
        "mode": "Diretto (senza proxy)",
    }
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True, verify=False) as client:
            resp_vix = await client.get("https://vixsrc.to/cdn-cgi/trace", headers={"User-Agent": "Mozilla/5.0"})
            if resp_vix.status_code == 200:
                fonte1.update({
                    "status": "ok",
                    "code": 200,
                    "message": "Raggiungibile direttamente senza blocchi (200 OK)",
                })
            else:
                fonte1.update({
                    "status": "warning",
                    "code": resp_vix.status_code,
                    "message": f"Risposta HTTP {resp_vix.status_code}",
                })
    except Exception as exc:
        fonte1.update({
            "status": "error",
            "code": 0,
            "message": f"Errore di rete: {exc}",
        })

    # --- Fonte 2 (Usa WARP se configurato in SIDECAR_AUDIO_PROXY) ---
    fonte2 = {
        "name": "Fonte 2",
        "proxy": proxy if proxy else "Diretto (nessun proxy)",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    kwargs = {"timeout": 10.0, "follow_redirects": True, "verify": False}
    if proxy:
        kwargs["proxy"] = proxy
    try:
        async with httpx.AsyncClient(**kwargs) as client:
            resp_pcc = await client.get("https://www.partite.cc/", headers=headers)
            if resp_pcc.status_code == 200:
                fonte2.update({
                    "status": "ok",
                    "code": 200,
                    "message": "Raggiungibile tramite WARP/Proxy (200 OK)" if proxy else "Raggiungibile direttamente senza blocchi (200 OK)",
                })
            else:
                msg = f"Risposta HTTP {resp_pcc.status_code}"
                if resp_pcc.status_code in (403, 503):
                    msg = f"Accesso filtrato o non autorizzato (HTTP {resp_pcc.status_code})"
                fonte2.update({
                    "status": "blocked" if not proxy else "warning",
                    "code": resp_pcc.status_code,
                    "message": msg,
                    "suggestion": (
                        "L'IP di questa macchina è filtrato per la Fonte 2. Configura Cloudflare WARP impostando "
                        "SIDECAR_AUDIO_PROXY=socks5h://172.17.0.1:1080 nel compose.yml (vedi guida nel README). "
                        "Nota: Fonte 1 continuerà a funzionare normalmente in connessione diretta."
                    ) if not proxy else (
                        "Verifica che il servizio proxy/WARP configurato in SIDECAR_AUDIO_PROXY sia attivo e funzionante."
                    ),
                })
    except Exception as exc:
        fonte2.update({
            "status": "error",
            "code": 0,
            "message": f"Errore di rete: {exc}",
            "suggestion": (
                "Impossibile raggiungere Fonte 2. Se usi una VPS con IP datacenter, "
                "configura Cloudflare WARP impostando SIDECAR_AUDIO_PROXY (vedi README)."
            ) if not proxy else (
                "Impossibile connettersi al proxy specificato in SIDECAR_AUDIO_PROXY. Verifica che sia attivo."
            ),
        })

    overall_status = "ok" if (fonte1.get("status") == "ok" and fonte2.get("status") == "ok") else "warning"
    return {
        "status": overall_status,
        "proxy": proxy if proxy else "Diretto (nessun proxy)",
        "fonte1": fonte1,
        "fonte2": fonte2,
    }


@app.post("/session")
async def create_session(request: Request):
    if BOOTSTRAP_KEY:
        supplied = request.headers.get("x-sidecar-bootstrap", "")
        if supplied != BOOTSTRAP_KEY:
            raise HTTPException(status_code=401, detail="bootstrap key required")
    sessions.cleanup()
    token, expires_at = sessions.issue()
    return {"token": token, "expires_at": expires_at, "ttl_seconds": sessions.ttl_seconds}


@app.post("/dual/aprep")
async def prepare_audio(request: Request):
    body = await request.json()
    token = _require_session(request, body)
    try:
        hid = await audio.register(
            playlist=str(body.get("playlist") or ""),
            key_b64=str(body.get("key") or ""),
            media_key=str(body.get("mediaKey") or ""),
            language=str(body.get("lang") or ""),
            base_url=str(body.get("baseUrl") or ""),
            headers=body.get("headers") if isinstance(body.get("headers"), dict) else {},
        )
        metadata = audio.metadata(hid)
        for url in (metadata["segs"][0], metadata["segs"][-1]):
            if not await resolves_publicly(url):
                raise ValueError("audio source does not resolve publicly")
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    language = str(body.get("lang") or "").lower()
    metadata = audio.metadata(hid)
    return JSONResponse({
        "hid": hid,
        "url": _audio_url(request, hid, token),
        "language": language,
        "audio_fingerprint": metadata.get("source_fingerprint", ""),
    })


@app.post("/dual/acache")
async def cached_audio(request: Request):
    body = await request.json()
    token = _require_session(request, body)
    hid = audio.find_cached(str(body.get("mediaKey") or ""), str(body.get("lang") or "").lower())
    if not hid:
        raise HTTPException(status_code=404, detail="valid cached audio track not found")
    metadata = audio.metadata(hid)
    return {
        "url": _audio_url(request, hid, token),
        "cached": True,
        "hid": hid,
        "audio_fingerprint": metadata.get("source_fingerprint", ""),
    }


def _audio_response(path: Path, media_type: str, cache_control: str = "no-cache"):
    return FileResponse(path, media_type=media_type, headers={
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": cache_control,
        "Accept-Ranges": "bytes",
    })


@app.api_route("/dual/aud/{hid}/audio.m3u8", methods=["GET", "HEAD"])
async def audio_playlist(hid: str, request: Request, o: int = 0, r: int = 1_000_000_000,
                         c: str = "", b: str = ""):
    token = _require_session(request)
    try:
        metadata = audio.metadata(hid)
        offset, rate = o / 1000.0, r / 1_000_000_000
        cuts = parse_cuts_param(c) if c else None
        bridge_hid = b.strip() if b else ""
        bridge_metadata = None
        if bridge_hid:
            try:
                bridge_metadata = audio.metadata(bridge_hid)
            except Exception:
                bridge_metadata = None

        timeline = audio.timeline(
            metadata, offset, rate, cuts=cuts, bridge_metadata=bridge_metadata,
            hid=hid, bridge_hid=bridge_hid
        )
        if not timeline:
            raise ValueError("empty audio timeline")

        base = _base_url(request)
        target_duration = int(max(item["duration"] for item in timeline)) + 1
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:7",
            "#EXT-X-PLAYLIST-TYPE:VOD",
            f"#EXT-X-TARGETDURATION:{target_duration}",
            "#EXT-X-MEDIA-SEQUENCE:0",
        ]

        current_map_hid = None
        for item in timeline:
            item_hid = item.get("hid") or hid
            item_query = {"o": o, "r": r, "t": token}
            if c:
                item_query["c"] = c
            if b:
                item_query["b"] = b
            q_str = urlencode(item_query)

            if item.get("discontinuity"):
                lines.append("#EXT-X-DISCONTINUITY")

            if current_map_hid != item_hid:
                current_map_hid = item_hid
                lines.append(f'#EXT-X-MAP:URI="{base}/dual/aud/{item_hid}/init.mp4?{q_str}"')

            lines.append(f"#EXTINF:{item['duration']:.6f},")
            lines.append(f"{base}/dual/aud/{item_hid}/s{item['idx']}.m4s?{q_str}")

        lines.append("#EXT-X-ENDLIST")
        return Response("\n".join(lines) + "\n", media_type="application/vnd.apple.mpegurl",
                        headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-cache"})
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.api_route("/dual/aud/{hid}/init.mp4", methods=["GET", "HEAD"])
async def audio_init(hid: str, request: Request, o: int = 0, r: int = 1_000_000_000,
                     c: str = "", b: str = ""):
    _require_session(request)
    try:
        metadata = audio.metadata(hid)
        cuts = parse_cuts_param(c) if c else None
        bridge_hid = b.strip() if b else ""
        bridge_metadata = audio.metadata(bridge_hid) if bridge_hid else None
        timeline = audio.timeline(metadata, o / 1000.0, r / 1_000_000_000, cuts=cuts,
                                  bridge_metadata=bridge_metadata, hid=hid, bridge_hid=bridge_hid)
        if not timeline:
            raise ValueError("empty audio timeline")
        first_seg = next((item["idx"] for item in timeline if item.get("hid", hid) == hid), 0)
        init_path, _ = await audio.fragment(hid, first_seg, o / 1000.0, r / 1_000_000_000,
                                            cuts=cuts, bridge_metadata=bridge_metadata)
        return _audio_response(init_path, "video/mp4", "public, max-age=3600")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.api_route("/dual/aud/{hid}/s{idx}.m4s", methods=["GET", "HEAD"])
async def audio_segment(hid: str, idx: int, request: Request, o: int = 0, r: int = 1_000_000_000,
                        c: str = "", b: str = ""):
    _require_session(request)
    try:
        cuts = parse_cuts_param(c) if c else None
        bridge_hid = b.strip() if b else ""
        bridge_metadata = audio.metadata(bridge_hid) if bridge_hid else None
        _, fragment_path = await audio.fragment(hid, idx, o / 1000.0, r / 1_000_000_000,
                                                cuts=cuts, bridge_metadata=bridge_metadata)
        return _audio_response(fragment_path, "video/iso.segment", "public, max-age=3600")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/offset/lookup")
async def offset_lookup(request: Request):
    body = await request.json()
    _require_session(request, body)
    result = await offsets.lookup(body)
    return {"found": bool(result), "offset": result}


@app.post("/offset/report")
async def offset_report(request: Request):
    body = await request.json()
    _require_session(request, body)
    result = body.get("offset")
    if not isinstance(result, dict):
        raise HTTPException(status_code=400, detail="offset result required")
    await offsets.report(body, result)
    return {"ok": True}


@app.post("/sync")
async def sync_audio(request: Request):
    body = await request.json()
    _require_session(request, body)
    try:
        result = await sync_engine.measure(body)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        err_msg = str(exc)
        if "audio segment fetch failed" in err_msg or "media fetch failed" in err_msg:
            return JSONResponse(
                status_code=502,
                content={"status": "audio_fetch_failed", "error": err_msg}
            )
        raise HTTPException(status_code=422, detail=str(exc))
    if result.get("status") in ("in_progress", "sync_in_progress") or result.get("background_sync"):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "SYNC_IN_PROGRESS",
                "message": result.get("message") or "⚠️ Sincronizzazione approfondita in corso in background. Riprova tra 15-20 secondi.",
            },
        )
    await offsets.report(body, result)
    if result.get("status") != "ok":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "EDITIONS_DIFFERENT",
                "message": "Edizioni audio e video differenti: sincronizzazione affidabile impossibile.",
            },
        )
    return result
