
#!/usr/bin/env python3
import os
import sqlite3
import json
import datetime
import threading
import time
import random
import urllib.request
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Single-line instantiation to guarantee clean global compilation scope
app = FastAPI()

DB_PATH = "local_anpr.db"
CONFIG_PATH = "edge_config.json"

# ================================================================================
# DEFAULT HARDWARE CONFIGURATION PROFILE
# ================================================================================
DEFAULT_CONFIG = {
    "cam1_enabled": True,
    "cam1_url": "rtsp://admin:securepass1@192.168.1.55:554/h264Preview_01_sub",
    "cam2_enabled": True,
    "cam2_url": "rtsp://admin:securepass1@192.168.1.56:554/h264Preview_01_sub",
    "server_sync_enabled": True,
    "server_url": "http://localhost:8000/api/v1/anpr/ingest",
    "server_token": "hub-bham-north"
}

def load_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except:
        return DEFAULT_CONFIG

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)

# Load layout parameters initially
current_config = load_config()


# ================================================================================
# DATABASE SETUP AND INITIALIZATION
# ================================================================================
def init_local_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS local_plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_name TEXT NOT NULL,
            plate TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL,
            sync_status TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

init_local_db()


# ================================================================================
# PROGRAMMATIC BACKGROUND SYNCHRONIZATION AND CAMERA SIMULATOR RUNTIMES
# ================================================================================
def transmit_payload_to_central_server(url, token, plate, confidence, timestamp):
    """Dispatches a single plate packet to the cloud engine using standard library modules."""
    payload = {
        "plate": plate,
        "confidence": float(confidence),
        "timestamp": timestamp
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    
    # 3 second maximum connection timeout to prevent hanging the edge detection thread
    with urllib.request.urlopen(req, timeout=3) as response:
        return response.status == 200 or response.status == 201


def background_cache_sync_worker():
    """Sweeps through the local SQLite table partitions to flush unsent offline traces."""
    global current_config
    while True:
        time.sleep(5)
        if not current_config.get("server_sync_enabled"):
            continue
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, plate, confidence, timestamp FROM local_plates WHERE sync_status = 'Pending'")
        pending_rows = cursor.fetchall()
        conn.close()
        
        if not pending_rows:
            continue
            
        for row in pending_rows:
            row_id, plate, conf, ts = row
            try:
                success = transmit_payload_to_central_server(
                    current_config["server_url"],
                    current_config["server_token"],
                    plate, conf, ts
                )
                if success:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE local_plates SET sync_status = 'Sent' WHERE id = ?", (row_id,))
                    conn.commit()
                    conn.close()
            except Exception:
                # Connection dropped or server down; will retry automatically during the next cycle loop
                break


def camera_anpr_inference_simulator():
    """Simulates active NPU classification logic loops processing frames from camera sub-streams."""
    global current_config
    mock_plates = ["GB26 XTR", "LS11 MPO", "WM76 KPL", "BHM 015B", "RE08 LNK", "OPI 4PRO", "AZ99 TWT", "NX55 VME"]
    
    while True:
        # Generate an automated traffic capture frame every 7 to 15 seconds
        time.sleep(random.randint(7, 15))
        
        active_feeds = []
        if current_config.get("cam1_enabled"): active_feeds.append("Cam 1: North Gate")
        if current_config.get("cam2_enabled"): active_feeds.append("Cam 2: South Exit")
        
        if not active_feeds:
            continue
            
        target_camera = random.choice(active_feeds)
        selected_plate = random.choice(mock_plates)
        calculated_confidence = round(random.uniform(92.0, 99.8), 1)
        current_time_str = datetime.datetime.now().isoformat()
        
        # Determine tracking status initial label
        initial_sync_state = "Pending"
        
        # Write directly to local storage layout first (Mandatory requirement)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO local_plates (camera_name, plate, confidence, timestamp, sync_status)
            VALUES (?, ?, ?, ?, ?);
        """, (target_camera, selected_plate, calculated_confidence, current_time_str, initial_sync_state))
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # If immediate upload is enabled, attempt instant network transmission
        if current_config.get("server_sync_enabled"):
            try:
                success = transmit_payload_to_central_server(
                    current_config["server_url"],
                    current_config["server_token"],
                    selected_plate, calculated_confidence, current_time_str
                )
                if success:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE local_plates SET sync_status = 'Sent' WHERE id = ?", (row_id,))
                    conn.commit()
                    conn.close()
            except Exception:
                # Synchronization failed; remains cleanly flagged as Pending for the background queue worker
                pass

# Initialize edge daemon runtimes asynchronously
threading.Thread(target=background_cache_sync_worker, daemon=True).start()
threading.Thread(target=camera_anpr_inference_simulator, daemon=True).start()


# ================================================================================
# REST PYDANTIC SCHEMAS AND INTERFACE CONTROL OBJECTS
# ================================================================================
class EdgeConfigSchema(BaseModel):
    cam1_enabled: bool
    cam1_url: str
    cam2_enabled: bool
    cam2_url: str
    server_sync_enabled: bool
    server_url: str
    server_token: str


@app.get("/api/config")
async def get_edge_configuration():
    global current_config
    return current_config


@app.post("/api/config")
async def update_edge_configuration(payload: EdgeConfigSchema):
    global current_config
    current_config = payload.dict()
    save_config(current_config)
    return {"status": "Success", "message": "Edge processing configuration synchronized."}


@app.get("/api/plates")
async def get_local_database_records():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, camera_name, plate, confidence, timestamp, sync_status FROM local_plates ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/telemetry")
async def get_system_telemetry_metrics():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM local_plates")
    total_records = cursor.fetchone()[0]
    conn.close()
    
    # Simulate LPDDR5 and Octa-Core system metric tracking profiles dynamically
    return {
        "cpu_load": f"{random.randint(12, 19)}%",
        "ram_usage": f"{round(random.uniform(1.3, 1.6), 1)} GB / 8 GB",
        "npu_util": f"{random.randint(20, 26)}% Utilization",
        "db_count": f"{total_records:,} Records"
    }


# ================================================================================
# MAIN PRODUCTION RE-RENDERING FOR INTEGRATED WEB INTERFACE
# ================================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_edge_administration_panel():
    # Production-ready Tailwind UI mapping directly to the endpoints built above
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Orange Pi 4 Pro - ANPR Configuration Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; background-color: #0f172a; }
        .scan-line { animation: scan 2.5s linear infinite; }
        @keyframes scan { 0% { top: 0%; } 50% { top: 100%; } 100% { top: 0%; } }
    </style>
</head>
<body class="text-slate-100 min-h-screen flex flex-col">

    <header class="bg-slate-900 border-b border-slate-800 px-6 py-4 sticky top-0 z-50 shadow-md">
        <div class="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
            <div class="flex items-center gap-3">
                <div class="bg-orange-600 text-white px-2.5 py-1 rounded-lg font-bold tracking-wider text-xs">OPI 4 PRO</div>
                <div>
                    <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                        EdgeANPR Hub 
                        <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Active</span>
                    </h1>
                    <p class="text-xs text-slate-400">Allwinner A733 octa-core edge terminal device</p>
                </div>
            </div>
            
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 bg-slate-950 p-2.5 rounded-xl border border-slate-800 text-xs w-full md:w-auto">
                <div><div class="text-slate-400 font-medium px-2">CPU Load</div><div class="text-slate-200 font-bold px-2" id="cpu-stat">--</div></div>
                <div class="border-l border-slate-800"><div class="text-slate-400 font-medium px-2">LPDDR5 RAM</div><div class="text-slate-200 font-bold px-2" id="ram-stat">--</div></div>
                <div class="border-l border-slate-800"><div class="text-slate-400 font-medium px-2">3 TOPS NPU</div><div class="text-slate-200 font-bold px-2" id="npu-stat">--</div></div>
                <div class="border-l border-slate-800"><div class="text-slate-400 font-medium px-2">Local Database</div><div class="text-slate-200 font-bold px-2 text-orange-400" id="storage-stat">--</div></div>
            </div>
        </div>
    </header>

    <nav class="bg-slate-900/60 border-b border-slate-800/80 px-6 py-2.5">
        <div class="max-w-7xl mx-auto flex gap-2">
            <button id="tab-btn-config" onclick="switchTab('config')" class="text-xs font-semibold px-4 py-2 rounded-xl transition bg-orange-600 text-white shadow shadow-orange-600/10">Configuration Console</button>
            <button id="tab-btn-db" onclick="switchTab('db')" class="text-xs font-semibold px-4 py-2 rounded-xl transition text-slate-400 hover:text-white hover:bg-slate-800/50">Detected plates</button>
        </div>
    </nav>

    <main class="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6">
        <div id="toast" class="fixed bottom-5 right-5 bg-emerald-600 text-white px-5 py-3 rounded-xl shadow-2xl transition-all duration-300 transform translate-y-24 opacity-0 pointer-events-none z-50 border border-emerald-400/20">
            <span id="toast-msg" class="font-medium text-sm">Settings saved!</span>
        </div>

        <!-- CONFIGURATION WORKSPACE -->
        <div id="tab-view-config" class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <section class="lg:col-span-7 flex flex-col gap-6">
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-sm relative flex flex-col">
                        <div class="p-3 bg-slate-950/60 border-b border-slate-800 flex justify-between items-center">
                            <span class="text-xs font-semibold text-slate-300 uppercase flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-orange-500" id="cam1-dot"></span>Cam 1: North Gate</span>
                        </div>
                        <div class="bg-slate-950 aspect-video relative flex items-center justify-center overflow-hidden">
                            <div class="absolute inset-0 bg-cover bg-center opacity-40 mix-blend-luminosity scale-105" style="background-image: url('https://placehold.co/600x400/1e293b/475569?text=North+Gate+Approach');"></div>
                            <div class="absolute top-0 left-0 w-full h-0.5 bg-orange-500/50 scan-line"></div>
                            <div id="cam1-overlay" class="absolute bottom-3 left-3 right-3 bg-slate-900/90 backdrop-blur border border-slate-700/50 p-2.5 rounded-lg flex items-center justify-between opacity-0 transition-all duration-300">
                                <div><div class="text-[10px] text-orange-400 font-mono font-semibold">Detected Plate</div><div class="text-lg font-bold font-mono text-white" id="cam1-overlay-plate">--</div></div>
                                <div class="text-right"><div class="text-[10px] text-slate-400">Confidence</div><div class="text-sm font-bold text-emerald-400 font-mono" id="cam1-overlay-conf">0.0%</div></div>
                            </div>
                        </div>
                    </div>
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-sm relative flex flex-col">
                        <div class="p-3 bg-slate-950/60 border-b border-slate-800 flex justify-between items-center">
                            <span class="text-xs font-semibold text-slate-300 uppercase flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-orange-500" id="cam2-dot"></span>Cam 2: South Exit</span>
                        </div>
                        <div class="bg-slate-950 aspect-video relative flex items-center justify-center overflow-hidden">
                            <div class="absolute inset-0 bg-cover bg-center opacity-40 mix-blend-luminosity scale-105" style="background-image: url('https://placehold.co/600x400/1e293b/475569?text=South+Exit+Lane');"></div>
                            <div class="absolute top-0 left-0 w-full h-0.5 bg-orange-500/50 scan-line"></div>
                            <div id="cam2-overlay" class="absolute bottom-3 left-3 right-3 bg-slate-900/90 backdrop-blur border border-slate-700/50 p-2.5 rounded-lg flex items-center justify-between opacity-0 transition-all duration-300">
                                <div><div class="text-[10px] text-orange-400 font-mono font-semibold">Detected Plate</div><div class="text-lg font-bold font-mono text-white" id="cam2-overlay-plate">--</div></div>
                                <div class="text-right"><div class="text-[10px] text-slate-400">Confidence</div><div class="text-sm font-bold text-emerald-400 font-mono" id="cam2-overlay-conf">0.0%</div></div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col flex-1 min-h-[300px]">
                    <h3 class="text-sm font-bold text-white mb-3">Live Terminal Processing Feed</h3>
                    <div class="flex-1 overflow-y-auto border border-slate-950 bg-slate-950 rounded-xl p-3 font-mono text-[11px] leading-relaxed max-h-[300px]" id="logs-container">
                        <div class="text-slate-500">[SYSTEM] Local edge runtime engine online.</div>
                    </div>
                </div>
            </section>

            <section class="lg:col-span-5 flex flex-col gap-6">
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm flex flex-col gap-4">
                    <div>
                        <h2 class="text-base font-bold text-white">ANPR Edge Configuration</h2>
                        <p class="text-xs text-slate-400">Define hardware video pipelines and upload rules</p>
                    </div>
                    <hr class="border-slate-800">
                    <div class="flex flex-col gap-4">
                        <div>
                            <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Camera Feed 1</label>
                            <div class="flex items-center gap-2 mb-2"><input type="checkbox" id="cam1-enabled" class="w-4 h-4 rounded accent-orange-500 bg-slate-950 border-slate-800"><span class="text-xs text-slate-300">Enable Feed 1 Parsing</span></div>
                            <input type="text" id="cam1-url" class="w-full text-xs font-mono bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-200 focus:outline-none">
                        </div>
                        <div>
                            <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Camera Feed 2</label>
                            <div class="flex items-center gap-2 mb-2"><input type="checkbox" id="cam2-enabled" class="w-4 h-4 rounded accent-orange-500 bg-slate-950 border-slate-800"><span class="text-xs text-slate-300">Enable Feed 2 Parsing</span></div>
                            <input type="text" id="cam2-url" class="w-full text-xs font-mono bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-200 focus:outline-none">
                        </div>
                        <hr class="border-slate-800">
                        <div>
                            <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Local Storage Mode</label>
                            <div class="bg-slate-950 border border-slate-800 rounded-xl p-3 flex justify-between items-center">
                                <div><div class="text-xs font-medium text-slate-200">Local DB Storage</div><div class="text-[10px] text-slate-500">Always active to secure captures safely</div></div>
                                <span class="text-[10px] bg-emerald-500/10 text-emerald-400 font-bold border border-emerald-500/20 px-2.5 py-1 rounded-full uppercase tracking-wider">Mandatory</span>
                            </div>
                        </div>
                        <hr class="border-slate-800">
                        <div>
                            <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">Central Server Sync Pipeline</label>
                            <div class="bg-slate-950 border border-slate-800 rounded-xl p-4 flex flex-col gap-3">
                                <div class="flex items-center justify-between">
                                    <div><div class="text-xs font-semibold text-slate-200">Immediate Cloud Upload</div><div class="text-[10px] text-slate-400">Stream records to server instantly when available</div></div>
                                    <label class="relative inline-flex items-center cursor-pointer">
                                        <input type="checkbox" id="server-sync-enabled" class="sr-only peer">
                                        <div class="w-8 h-4 bg-slate-800 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-slate-400 after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-orange-500 peer-checked:after:bg-white"></div>
                                    </label>
                                </div>
                                <div class="flex flex-col gap-2 pt-2 border-t border-slate-900">
                                    <div><label class="block text-[10px] text-slate-400 font-medium mb-1">Central API Ingest Terminal URL</label><input type="url" id="server-url" class="w-full text-xs font-mono bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-300 focus:outline-none"></div>
                                    <div><label class="block text-[10px] text-slate-400 font-medium mb-1">Hub Account Token ID</label><input type="text" id="server-token" class="w-full text-xs font-mono bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-300 focus:outline-none"></div>
                                </div>
                            </div>
                        </div>
                        <div class="grid grid-cols-2 gap-3 mt-2">
                            <button onclick="loadEdgeConfig()" class="border border-slate-800 bg-slate-950 text-slate-300 font-medium text-xs px-4 py-2.5 rounded-xl transition">Discard</button>
                            <button onclick="saveEdgeConfigForm()" class="bg-orange-600 text-white font-semibold text-xs px-4 py-2.5 rounded-xl transition shadow-md">Apply Properties</button>
                        </div>
                    </div>
                </div>
            </section>
        </div>

        <!-- DATABASE WORKSPACE VIEW -->
        <div id="tab-view-db" class="hidden bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm flex flex-col gap-4">
            <div class="flex flex-col sm:flex-row justify-between sm:items-center gap-2">
                <div>
                    <h2 class="text-base font-bold text-white">Detected Plates History</h2>
                    <p class="text-xs text-slate-400">Master database records saved on local hardware device</p>
                </div>
                <div class="flex gap-2">
                    <button onclick="sortDatabase('plate')" class="bg-slate-800 text-slate-300 px-3 py-1.5 rounded-xl text-xs hover:text-white font-medium transition">Sort by Reg</button>
                    <button onclick="sortDatabase('timestamp')" class="bg-slate-800 text-slate-300 px-3 py-1.5 rounded-xl text-xs hover:text-white font-medium transition">Sort by Date/Time</button>
                </div>
            </div>
            <div class="overflow-x-auto border border-slate-800 bg-slate-950 rounded-xl">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="bg-slate-900/60 border-b border-slate-800 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                            <th class="px-6 py-3">Timestamp Index</th>
                            <th class="px-6 py-3">Source Channel</th>
                            <th class="px-6 py-3">Plate Alphanumeric</th>
                            <th class="px-6 py-3">NPU Confidence</th>
                            <th class="px-6 py-3">Cloud Status</th>
                        </tr>
                    </thead>
                    <tbody id="db-table-body" class="divide-y divide-slate-800/40">
                        <!-- Placed programmatically via internal API arrays -->
                    </tbody>
                </table>
            </div>
        </div>
    </main>

    <script>
        let currentTab = "config";
        let dbCache = [];
        let sortDirections = { plate: 1, timestamp: -1 };
        let visualFlashCache = {};

        function switchTab(tabId) {
            currentTab = tabId;
            document.getElementById("tab-view-config").classList.toggle("hidden", tabId !== "config");
            document.getElementById("tab-view-db").classList.toggle("hidden", tabId !== "db");
            
            document.getElementById("tab-btn-config").className = tabId === "config" ? "text-xs font-semibold px-4 py-2 rounded-xl transition bg-orange-600 text-white shadow" : "text-xs font-semibold px-4 py-2 rounded-xl transition text-slate-400 hover:text-white hover:bg-slate-800/50";
            document.getElementById("tab-btn-db").className = tabId === "db" ? "text-xs font-semibold px-4 py-2 rounded-xl transition bg-orange-600 text-white shadow" : "text-xs font-semibold px-4 py-2 rounded-xl transition text-slate-400 hover:text-white hover:bg-slate-800/50";
            
            if (tabId === "db") refreshDatabaseTable();
        }

        async function loadEdgeConfig() {
            let res = await fetch("/api/config");
            let data = await res.json();
            document.getElementById("cam1-enabled").checked = data.cam1_enabled;
            document.getElementById("cam1-url").value = data.cam1_url;
            document.getElementById("cam2-enabled").checked = data.cam2_enabled;
            document.getElementById("cam2-url").value = data.cam2_url;
            document.getElementById("server-sync-enabled").checked = data.server_sync_enabled;
            document.getElementById("server-url").value = data.server_url;
            document.getElementById("server-token").value = data.server_token;
        }

        async function saveEdgeConfigForm() {
            let configPayload = {
                cam1_enabled: document.getElementById("cam1-enabled").checked,
                cam1_url: document.getElementById("cam1-url").value,
                cam2_enabled: document.getElementById("cam2-enabled").checked,
                cam2_url: document.getElementById("cam2-url").value,
                server_sync_enabled: document.getElementById("server-sync-enabled").checked,
                server_url: document.getElementById("server-url").value,
                server_token: document.getElementById("server-token").value
            };
            let res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: json = JSON.stringify(configPayload)
            });
            if (res.ok) {
                showToastNotification("Edge parameters saved and reloaded!");
                appendLogMessage("[CONFIG] Core system settings configuration overwritten successfully.");
            }
        }

        async function updateTelemetryAndFeeds() {
            let res = await fetch("/api/telemetry");
            let metrics = await res.json();
            document.getElementById("cpu-stat").innerText = metrics.cpu_load;
            document.getElementById("ram-stat").innerText = metrics.ram_usage;
            document.getElementById("npu-stat").innerText = metrics.npu_util;
            document.getElementById("storage-stat").innerText = metrics.db_count;

            let platesRes = await fetch("/api/plates");
            let plates = await platesRes.json();
            dbCache = plates;
            
            if (plates.length > 0 && currentTab === "config") {
                let latestItem = plates[0];
                if (!visualFlashCache[latestItem.id]) {
                    visualFlashCache[latestItem.id] = true;
                    triggerOverlayFlash(latestItem);
                }
            }
            if (currentTab === "db") renderTableRows(dbCache);
        }

        function triggerOverlayFlash(item) {
            let targetId = item.camera_name.includes("Cam 1") ? 1 : 2;
            let overlay = document.getElementById(`cam${targetId}-overlay`);
            document.getElementById(`cam${targetId}-overlay-plate`).innerText = item.plate;
            document.getElementById(`cam${targetId}-overlay-conf`).innerText = `${item.confidence}%`;
            
            overlay.classList.remove("opacity-0");
            overlay.classList.add("opacity-100");
            
            appendLogMessage(`[NPU INFERENCE] ${item.camera_name} - Isolated Plate: ${item.plate} (${item.confidence}%) -> Saved to local database. Sync Status: ${item.sync_status}`);
            
            setTimeout(() => { overlay.classList.remove("opacity-100"); overlay.classList.add("opacity-0"); }, 3500);
        }

        function renderTableRows(items) {
            let tbody = document.getElementById("db-table-body");
            tbody.innerHTML = "";
            items.forEach(row => {
                let formattedTime = row.timestamp.replace("T", " ").split(".")[0];
                let syncBadge = row.sync_status === "Sent" 
                    ? `<span class="bg-emerald-500/10 text-emerald-400 text-[10px] px-2 py-0.5 rounded border border-emerald-500/20 font-bold">Sent</span>`
                    : `<span class="bg-amber-500/10 text-amber-400 text-[10px] px-2 py-0.5 rounded border border-amber-500/20 font-bold animate-pulse">Pending</span>`;
                
                let tr = document.createElement("tr");
                tr.className = "border-b border-slate-800/60 hover:bg-slate-800/20 text-xs transition-colors";
                tr.innerHTML = `
                    <td class="px-6 py-3 text-slate-400 font-mono">${formattedTime}</td>
                    <td class="px-6 py-3 text-slate-300 font-sans">${row.camera_name}</td>
                    <td class="px-6 py-3"><span class="bg-slate-950 px-2 py-1 rounded border border-slate-800 text-white font-mono font-bold font-tracking-wider">${row.plate}</span></td>
                    <td class="px-6 py-3 font-mono text-emerald-400 font-medium">${row.confidence}%</td>
                    <td class="px-6 py-3">${syncBadge}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function sortDatabase(field) {
            sortDirections[field] = -sortDirections[field];
            let direction = sortDirections[field];
            dbCache.sort((a, b) => {
                if (a[field] < b[field]) return -direction;
                if (a[field] > b[field]) return direction;
                return 0;
            });
            renderTableRows(dbCache);
        }

        function appendLogMessage(msg) {
            let container = document.getElementById("logs-container");
            let div = document.createElement("div");
            div.className = "text-slate-400";
            div.innerHTML = `<span class="text-slate-600">[${new Date().toLocaleTimeString()}]</span> ${msg}`;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        function showToastNotification(msg) {
            let toast = document.getElementById("toast");
            document.getElementById("toast-msg").innerText = msg;
            toast.classList.remove("opacity-0", "translate-y-24", "pointer-events-none");
            toast.classList.add("opacity-100", "translate-y-0");
            setTimeout(() => { toast.classList.remove("opacity-100", "translate-y-0"); toast.classList.add("opacity-0", "translate-y-24", "pointer-events-none"); }, 3000);
        }

        window.onload = function() {
            loadEdgeConfig();
            updateTelemetryAndFeeds();
            setInterval(updateTelemetryAndFeeds, 2500);
        };
    </script>
</body>
</html>
"""
    return html_content


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
