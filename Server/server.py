#!/usr/bin/env python3
import os
import sqlite3
import datetime
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Clean constructor instantiation
app = FastAPI()

# Persistent Engine Storage Configurations (No indentation requirements to prevent scope pitfalls)
os.makedirs("data", exist_ok=True)
DB_PATH = os.path.join("data", "central_anpr.db")


def init_db():
    """Initializes the SQLite schema tables if they are absent on local disk storage arrays."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Edge Hub Account Mapping Schema Partition
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hubs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    
    # 2. Aggregated Trace Metadata Table Log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hub_id TEXT NOT NULL,
            plate TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (hub_id) REFERENCES hubs(id) ON DELETE CASCADE
        );
    """)
    
    # Inject Initial Contextual Seed Data for evaluation on first run
    cursor.execute("SELECT COUNT(*) FROM hubs")
    if cursor.fetchone()[0] == 0:
        now_string = datetime.datetime.now().isoformat()
        
        # Seed example remote Hub nodes
        cursor.execute("INSERT INTO hubs VALUES (?, ?, ?, ?, ?)", ("hub-bham-north", "Birmingham North Gate", 52.4862, -1.8904, now_string))
        cursor.execute("INSERT INTO hubs VALUES (?, ?, ?, ?, ?)", ("hub-london-m25", "London Orbital Checkpoint", 51.5074, -0.1278, now_string))
        
        # Seed matching historical records matching August 2026 contexts
        cursor.execute("INSERT INTO plates (hub_id, plate, confidence, timestamp) VALUES (?, ?, ?, ?)", ("hub-bham-north", "GB26 XTR", 94.5, "2026-08-15T12:15:00"))
        cursor.execute("INSERT INTO plates (hub_id, plate, confidence, timestamp) VALUES (?, ?, ?, ?)", ("hub-london-m25", "RE08 LNK", 98.2, "2026-08-15T12:44:10"))
        cursor.execute("INSERT INTO plates (hub_id, plate, confidence, timestamp) VALUES (?, ?, ?, ?)", ("hub-bham-north", "OPI 4PRO", 99.1, "2026-08-15T13:02:15"))
        
    conn.commit()
    conn.close()


# Fire Schema Allocation Routines on application loader cycle
init_db()


# ================================================================================
# PYDANTIC VALIDATION API INTERACTION SCHEMAS
# ================================================================================

class HubRegistrationSchema(BaseModel):
    id: str  # Uniquely assigned identification key token (e.g., 'hub-bham-north')
    name: str  # Human-readable label for dashboard sorting metrics
    latitude: float  # Geographical latitude coordinate point mapping
    longitude: float  # Geographical longitude coordinate point mapping


class PlateIngestionSchema(BaseModel):
    plate: str  # Isolated alphanumeric plate character sequence
    confidence: float  # Deep neural network precision ranking float percentage
    timestamp: str  # Transaction isolation standard time string issued by node


# ================================================================================
# REST ENDPOINT INTERFACE ROUTING MATRIX
# ================================================================================

@app.post("/api/v1/hubs/register")
async def register_hub_account(payload: HubRegistrationSchema):
    """Registers a new decentralized Orange Pi Edge Hub account parameters."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now_string = datetime.datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO hubs (id, name, latitude, longitude, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                latitude = excluded.latitude,
                longitude = excluded.longitude;
        """, (payload.id, payload.name, payload.latitude, payload.longitude, now_string))
        
        conn.commit()
        conn.close()
        return JSONResponse(status_code=201, content={"status": "Success", "message": f"Account profile mapping locked for terminal ID: '{payload.id}'."})
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Database pipeline failure: {str(err)}")


@app.post("/api/v1/anpr/ingest")
async def ingest_plate_records(payload: PlateIngestionSchema, authorization: str = Header(None)):
    """Ingests plate telemetry packets dispatched directly by Orange Pi microcontrollers."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization bearer credentials header context.")
    
    extracted_hub_id = authorization.split(" ")[1]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM hubs WHERE id = ?", (extracted_hub_id,))
    hub_exists = cursor.fetchone()
    
    if not hub_exists:
        conn.close()
        raise HTTPException(status_code=403, detail=f"Access denied: No active server account profile allocated for token ID '{extracted_hub_id}'.")
    
    try:
        cursor.execute("""
            INSERT INTO plates (hub_id, plate, confidence, timestamp)
            VALUES (?, ?, ?, ?);
        """, (extracted_hub_id, payload.plate.upper().strip(), payload.confidence, payload.timestamp))
        
        conn.commit()
        conn.close()
        return {"status": "Ingested", "target": extracted_hub_id, "processed_item": payload.plate}
    except Exception as err:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Write sequence failed down-stream transaction: {str(err)}")


# ================================================================================
# EMBEDDED REAL-TIME SERVER ADMINISTRATIVE GUI MONITOR PANEL
# ================================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_admin_dashboard_console():
    """Compiles and streams a dense real-time dashboard layout monitoring global account states."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as total FROM hubs")
    hub_count = cursor.fetchone()['total']
    
    cursor.execute("SELECT COUNT(*) as total FROM plates")
    plate_count = cursor.fetchone()['total']
    
    cursor.execute("""
        SELECT h.id, h.name, h.latitude, h.longitude, COUNT(p.id) as volumes 
        FROM hubs h 
        LEFT JOIN plates p ON h.id = p.hub_id 
        GROUP BY h.id 
        ORDER BY h.name ASC
    """)
    hubs_rows = cursor.fetchall()
    
    cursor.execute("""
        SELECT p.timestamp, p.plate, p.confidence, h.name as hub_name 
        FROM plates p 
        JOIN hubs h ON p.hub_id = h.id 
        ORDER BY p.timestamp DESC, p.id DESC 
        LIMIT 50
    """)
    plates_rows = cursor.fetchall()
    conn.close()

    hubs_table_html = ""
    for hub in hubs_rows:
        hubs_table_html += f"""
        <tr class="border-b border-slate-800 hover:bg-slate-800/20 font-mono text-xs text-slate-300 transition-colors">
            <td class="px-6 py-4 font-bold text-orange-400">{hub['id']}</td>
            <td class="px-6 py-4 font-sans text-slate-200">{hub['name']}</td>
            <td class="px-6 py-4 text-cyan-400">{hub['latitude']:.4f}, {hub['longitude']:.4f}</td>
            <td class="px-6 py-4 font-bold text-slate-400">{hub['volumes']:,} rows</td>
        </tr>
        """

    plates_table_html = ""
    for item in plates_rows:
        clean_time = item['timestamp'].replace("T", " ")[:19]
        conf_val = float(item['confidence'])
        conf_color = "text-emerald-400 font-bold" if conf_val > 95.0 else "text-slate-300 font-medium"
        
        plates_table_html += f"""
        <tr class="border-b border-slate-800/60 hover:bg-slate-800/30 font-mono text-xs text-slate-300 transition-colors">
            <td class="px-6 py-3.5 text-slate-400">{clean_time}</td>
            <td class="px-6 py-3.5 font-sans text-slate-200">{item['hub_name']}</td>
            <td class="px-6 py-3.5"><span class="bg-slate-900 border border-slate-800 px-2 py-1 rounded text-white font-mono font-bold tracking-wider">{item['plate']}</span></td>
            <td class="px-6 py-3.5 {conf_color}">{conf_val:.1f}%</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Central Headquarters - ANPR Central Broker</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body {{ font-family: 'Inter', sans-serif; background-color: #0f172a; }}
    </style>
</head>
<body class="text-slate-100 min-h-screen flex flex-col">
    <header class="bg-slate-900 border-b border-slate-800 px-6 py-5 sticky top-0 z-50 shadow-lg shadow-slate-950/20">
        <div class="max-w-7xl mx-auto flex justify-between items-center">
            <div>
                <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                    Central ANPR Management HQ
                    <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">Broker Core</span>
                </h1>
                <p class="text-xs text-slate-400 mt-0.5">Distributed network receiver array dashboard console</p>
            </div>
            <div class="text-xs font-mono bg-slate-950 border border-slate-800 px-4 py-2 rounded-xl text-slate-400">
                System Epoch Time: <span class="text-slate-200 font-bold">2026-08-15</span>
            </div>
        </div>
    </header>
    <main class="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 flex flex-col gap-6">
        <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            <div class="bg-slate-900 border border-slate-800 p-5 rounded-2xl flex flex-col gap-1 shadow-sm">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Registered Terminal Hubs</div>
                <div class="text-2xl font-black text-orange-500 font-mono mt-1">{hub_count} Accounts</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 p-5 rounded-2xl flex flex-col gap-1 shadow-sm">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Aggregated Global Ingest</div>
                <div class="text-2xl font-black text-emerald-400 font-mono mt-1">{plate_count:,} Total</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 p-5 rounded-2xl flex flex-col gap-1 shadow-sm sm:col-span-2 md:col-span-1">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Broker Access Endpoint</div>
                <div class="text-xs font-mono text-cyan-400 mt-2 bg-slate-950 px-2.5 py-1.5 rounded-lg border border-slate-800/80 break-all">/api/v1/anpr/ingest</div>
            </div>
        </div>
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            <div class="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm flex flex-col gap-4">
                <div>
                    <h2 class="text-sm font-bold text-white tracking-tight">Active Edge Hub Nodes</h2>
                    <p class="text-xs text-slate-400 mt-0.5">Orange Pi hardware authentications and GPS location parameters</p>
                </div>
                <div class="overflow-x-auto border border-slate-800 bg-slate-950 rounded-xl">
                    <table class="w-full text-left border-collapse">
                        <thead>
                            <tr class="bg-slate-900 border-b border-slate-800 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                                <th class="px-6 py-3">Hub ID Token</th>
                                <th class="px-6 py-3">Location Name</th>
                                <th class="px-6 py-3">GPS Tracking</th>
                                <th class="px-6 py-3">Ingested</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800/40">
                            {hubs_table_html}
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm flex flex-col gap-4">
                <div>
                    <h2 class="text-sm font-bold text-white tracking-tight">Master Real-Time Synchronized Streams</h2>
                    <p class="text-xs text-slate-400 mt-0.5">Aggregated ingestion chronological ledger from all edge network segments</p>
                </div>
                <div class="overflow-x-auto border border-slate-800 bg-slate-950 rounded-xl">
                    <table class="w-full text-left border-collapse">
                        <thead>
                            <tr class="bg-slate-900 border-b border-slate-800 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                                <th class="px-6 py-3.5">Log Timestamp</th>
                                <th class="px-6 py-3.5">Source Terminal Node</th>
                                <th class="px-6 py-3.5">Plate Value</th>
                                <th class="px-6 py-3.5">NPU Confidence</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800/40">
                            {plates_table_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </main>
    <footer class="border-t border-slate-800 bg-slate-950/40 px-6 py-4 mt-12 text-center text-[11px] text-slate-500 font-mono">
        EdgeANPR Aggregator Core Node Engine Running inside Docker Sandbox.
    </footer>
</body>
</html>"""
    return html_content


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
