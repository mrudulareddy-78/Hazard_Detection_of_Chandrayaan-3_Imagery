
import os
from typing import List
from urllib.parse import urlparse

import mysql.connector
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

MYSQL_URL = os.getenv("MYSQL_URL")


def _parse_mysql_url(url: str) -> dict:
    parsed = urlparse(url)
    if parsed.scheme not in {"mysql", "mysql+mysqlconnector"}:
        raise ValueError("MYSQL_URL must start with mysql://")
    return {
        "user": parsed.username,
        "password": parsed.password,
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 3306,
        "database": (parsed.path or "/").lstrip("/")
    }


def _get_connection():
    if not MYSQL_URL:
        raise RuntimeError("MYSQL_URL is not set")
    config = _parse_mysql_url(MYSQL_URL)
    return mysql.connector.connect(**config)


def _init_db():
    try:
        conn = _get_connection()
    except Exception:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp VARCHAR(32) NOT NULL,
                safe FLOAT NOT NULL,
                rocks FLOAT NOT NULL,
                crater FLOAT NOT NULL,
                source VARCHAR(64) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS path_runs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp VARCHAR(32) NOT NULL,
                image_name VARCHAR(255),
                algorithm VARCHAR(64) NOT NULL,
                safety_mode BOOLEAN NOT NULL,
                start_row INT NOT NULL,
                start_col INT NOT NULL,
                goal_row INT NOT NULL,
                goal_col INT NOT NULL,
                planning_time_ms FLOAT,
                nodes_explored INT,
                path_length INT,
                total_cost FLOAT,
                safe_percentage FLOAT,
                risk_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


class Payload(BaseModel):
    timestamp: str
    safe: float
    rocks: float
    crater: float
    source: str


class PathRun(BaseModel):
    timestamp: str
    image_name: str | None = None
    algorithm: str
    safety_mode: bool
    start_row: int
    start_col: int
    goal_row: int
    goal_col: int
    planning_time_ms: float | None = None
    nodes_explored: int | None = None
    path_length: int | None = None
    total_cost: float | None = None
    safe_percentage: float | None = None
    risk_score: float | None = None


@app.post("/update")
def update(data: Payload):
    try:
        conn = _get_connection()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO telemetry (timestamp, safe, rocks, crater, source)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (data.timestamp, data.safe, data.rocks, data.crater, data.source)
        )
        conn.commit()
    finally:
        conn.close()

    return {"status": "received"}


@app.post("/path_run")
def path_run(data: PathRun):
    try:
        conn = _get_connection()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO path_runs (
                timestamp, image_name, algorithm, safety_mode,
                start_row, start_col, goal_row, goal_col,
                planning_time_ms, nodes_explored, path_length,
                total_cost, safe_percentage, risk_score
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                data.timestamp,
                data.image_name,
                data.algorithm,
                data.safety_mode,
                data.start_row,
                data.start_col,
                data.goal_row,
                data.goal_col,
                data.planning_time_ms,
                data.nodes_explored,
                data.path_length,
                data.total_cost,
                data.safe_percentage,
                data.risk_score
            )
        )
        conn.commit()
    finally:
        conn.close()

    return {"status": "received"}


@app.get("/latest")
def latest():
    try:
        conn = _get_connection()
    except Exception:
        return {}
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return row or {}
    finally:
        conn.close()


@app.get("/history")
def history():
    try:
        conn = _get_connection()
    except Exception:
        return []
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT 200")
        rows = cursor.fetchall()
        return list(reversed(rows))
    finally:
        conn.close()


@app.get("/path_runs")
def path_runs():
    try:
        conn = _get_connection()
    except Exception:
        return []
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM path_runs ORDER BY id DESC LIMIT 200")
        rows = cursor.fetchall()
        return list(reversed(rows))
    finally:
        conn.close()


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


@app.on_event("startup")
def _startup():
    _init_db()
