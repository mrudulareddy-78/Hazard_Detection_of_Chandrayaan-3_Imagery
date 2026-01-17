
from typing import List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

# In-memory time-series storage (ordered).
telemetry_history: List[dict] = []


class Payload(BaseModel):
    timestamp: str
    safe: float
    rocks: float
    crater: float
    source: str


@app.post("/update")
def update(data: Payload):
    record = data.dict()
    telemetry_history.append(record)
    return {"status": "received"}


@app.get("/latest")
def latest():
    return telemetry_history[-1] if telemetry_history else {}


@app.get("/history")
def history():
    return telemetry_history


@app.get("/", response_class=HTMLResponse)
def dashboard():
    # Lightweight dashboard with auto-refresh table sourced from /history
    html = """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
        <title>Hazard Telemetry Dashboard</title>
        <style>
            :root {
                --bg: #0b1021;
                --card: #121a33;
                --text: #e8edf7;
                --muted: #8ea0c6;
                --accent: #5dd5ff;
            }
            body { margin: 0; font-family: "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
            header { padding: 1.5rem 2rem; background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.06); }
            h1 { margin: 0; font-size: 1.6rem; letter-spacing: 0.02em; }
            main { padding: 1.5rem 2rem; }
            .card { background: var(--card); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 1rem; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
            table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
            th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.08); }
            th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }
            tr:hover { background: rgba(255,255,255,0.03); }
            .pill { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; background: rgba(93,213,255,0.12); color: var(--accent); font-size: 0.85rem; }
            footer { color: var(--muted); margin-top: 0.75rem; font-size: 0.9rem; }
        </style>
    </head>
    <body>
        <header>
            <h1>☾ Lunar Hazard Telemetry</h1>
            <div style=\"color: var(--muted); margin-top: 0.3rem;\">Edge → Cloud time-series (offline-first)</div>
        </header>
        <main>
            <div class=\"card\">
                <div style=\"display:flex; justify-content:space-between; align-items:center;\">
                    <div style=\"color: var(--muted);\">Auto-refreshing every 5s</div>
                    <div id=\"status\" class=\"pill\">Connecting...</div>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Safe %</th>
                            <th>Rocks %</th>
                            <th>Crater %</th>
                            <th>Source</th>
                        </tr>
                    </thead>
                    <tbody id=\"telemetry-body\">
                        <tr><td colspan=\"5\">Loading...</td></tr>
                    </tbody>
                </table>
                <footer id=\"footer\"></footer>
            </div>
        </main>
        <script>
            const bodyEl = document.getElementById('telemetry-body');
            const statusEl = document.getElementById('status');
            const footerEl = document.getElementById('footer');

            async function fetchHistory() {
                try {
                    statusEl.textContent = 'Syncing';
                    const res = await fetch('/history');
                    const data = await res.json();

                    if (!Array.isArray(data) || data.length === 0) {
                        bodyEl.innerHTML = '<tr><td colspan="5">No telemetry received yet.</td></tr>';
                        statusEl.textContent = 'Idle';
                        footerEl.textContent = '';
                        return;
                    }

                    bodyEl.innerHTML = data.map(item => {
                        return `<tr>
                            <td>${item.timestamp || '-'}</td>
                            <td>${Number(item.safe || 0).toFixed(2)}</td>
                            <td>${Number(item.rocks || 0).toFixed(2)}</td>
                            <td>${Number(item.crater || 0).toFixed(2)}</td>
                            <td>${item.source || '-'}</td>
                        </tr>`;
                    }).join('');

                    statusEl.textContent = 'Live';
                    footerEl.textContent = `Total records: ${data.length}`;
                } catch (err) {
                    statusEl.textContent = 'Offline';
                }
            }

            fetchHistory();
            setInterval(fetchHistory, 5000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
