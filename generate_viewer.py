import os
import json
import warnings
import concurrent.futures
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie

warnings.filterwarnings('ignore')

# --- 1. DIRECTORIES & CONFIGURATION ---
OUTPUT_DIR = "output"
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Forecast steps: 0 to 240 hours every 12 hours (21 lead times)
FXX_STEPS = list(range(0, 241, 12))
ALL_CONTOURS = list(range(4980, 6001, 60))

# Fixed canvas dimensions
CANVAS_W, CANVAS_H = 1200, 800

# Operational Cycle Detector
now = datetime.utcnow()
if (now.hour == 1 and now.minute > 30) or (2 <= now.hour <= 12) or (now.hour == 13 and now.minute <= 30):
    base_date = now - timedelta(days=1)
    global_cycle = "12"
else:
    base_date = now - timedelta(days=1) if now.hour <= 1 else now
    global_cycle = "00"

global_init_time = base_date.strftime(f'%Y-%m-%d {global_cycle}:00')
print(f"📡 Global Model Baseline Cycle: {global_init_time}")

# CONUS Domain
lon_min, lon_max = -130.0, -60.0
lat_min, lat_max = 20.0, 55.0
grid_lats = np.arange(lat_min, lat_max + 0.5, 0.5)
grid_lons = np.arange(lon_min, lon_max + 0.5, 0.5)
target_lons, target_lats = np.meshgrid(grid_lons, grid_lats)

# --- 2. GENERATE STATIC BASEMAP ---
print("🗺️ Rendering static Cartopy basemap...")
fig = plt.figure(figsize=(12, 8), dpi=100)
ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0))
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor="#f8f9fa")
ax.add_feature(cfeature.OCEAN, facecolor="#eef2f5")
ax.add_feature(cfeature.COASTLINE, linewidth=1.1, edgecolor="#333333")
ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor="#333333")
ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#888888", linestyle=":")
ax.add_feature(cfeature.LAKES, facecolor="#eef2f5", edgecolor="#888888", linewidth=0.4)

fig.savefig(os.path.join(OUTPUT_DIR, "basemap.png"), dpi=100)
print("✅ Saved basemap.png")

# Instantiate PlateCarree to display transform for the GeoAxes
geo_trans = ccrs.PlateCarree()._as_mpl_transform(ax)


# --- 3. GRID STANDARDIZATION & VECTOR EXTRACTION ---
def standardize_grid(ds):
    if ds is None:
        return None
    if isinstance(ds, list):
        if len(ds) == 0:
            return None
        ds = ds[0]
    try:
        ds = ds.squeeze()
        var_key = list(ds.data_vars)[0]
        da = ds[var_key]
        if da.longitude.max() > 180:
            da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
        da = da.sortby(['latitude', 'longitude'])
        da_interp = da.interp(latitude=grid_lats, longitude=grid_lons, method='linear')
        arr = da_interp.values
        if np.nanmean(arr) > 20000:
            arr = arr / 9.80665
        return arr
    except Exception:
        return None

def extract_contours(grid):
    """Computes contours and converts them to pixel coordinates."""
    if grid is None or np.all(np.isnan(grid)):
        return {}
    try:
        cs = ax.contour(target_lons, target_lats, grid, levels=ALL_CONTOURS, transform=ccrs.PlateCarree())
        result = {}
        
        for level, segs in zip(cs.levels, cs.allsegs):
            if not segs:
                continue
            lvl_key = str(int(level))
            lvl_lines = []
            for s in segs:
                if len(s) < 2:
                    continue
                # Transform PlateCarree (lon, lat) to screen display coordinates
                disp_coords = geo_trans.transform(s)
                # Transform screen display to normalized figure coordinates [0, 1]
                fig_coords = fig.transFigure.inverted().transform(disp_coords)
                
                px = np.round(fig_coords[:, 0] * CANVAS_W).astype(int)
                py = np.round((1.0 - fig_coords[:, 1]) * CANVAS_H).astype(int)
                
                # Filter duplicate/adjacent points
                pts = []
                for x, y in zip(px, py):
                    # Keep point if inside or near canvas bounds
                    if -50 <= x <= CANVAS_W + 50 and -50 <= y <= CANVAS_H + 50:
                        if not pts or (abs(x - pts[-1][0]) >= 2 or abs(y - pts[-1][1]) >= 2):
                            pts.append([int(x), int(y)])
                            
                if len(pts) > 1:
                    lvl_lines.append(pts)
                    
            if lvl_lines:
                result[lvl_key] = lvl_lines

        # Cross-version safe contour cleanup
        if hasattr(cs, 'remove'):
            cs.remove()
        elif hasattr(cs, 'collections'):
            for coll in cs.collections:
                coll.remove()
                
        return result
    except Exception as e:
        print(f"⚠️ Warning in contour extraction: {e}")
        return {}


# --- 4. DATA FETCHERS ---
def fetch_gefs_lead(fxx):
    members = ["c00"] + [f"p{str(i).zfill(2)}" for i in range(1, 31)]
    def _worker(m):
        try:
            H = Herbie(global_init_time, model="gefs", product="atmos.5", member=m, fxx=fxx)
            ds = H.xarray(":HGT:500 mb:")
            return standardize_grid(ds)
        except Exception:
            return None
    grids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for res in executor.map(_worker, members):
            if res is not None and res.ndim == 2:
                grids.append(res)
    return grids

def fetch_ifs_ensemble(fxx):
    grids = []
    try:
        H = Herbie(global_init_time, model="ifs", product="enfo", fxx=fxx)
        ds_list = H.xarray(":gh:500")
        if not isinstance(ds_list, list):
            ds_list = [ds_list]
        for ds in ds_list:
            g = standardize_grid(ds)
            if g is not None:
                if g.ndim == 3:
                    for i in range(g.shape[0]):
                        grids.append(g[i])
                elif g.ndim == 2:
                    grids.append(g)
    except Exception:
        pass
    return grids

def fetch_aifs_ensemble(fxx):
    grids = []
    try:
        H = Herbie(global_init_time, model="aifs", product="enfo", fxx=fxx)
        for data_type in ["pf", "cf"]:
            try:
                ds_list = H.xarray(":z:500:", backend_kwargs={"filter_by_keys": {"dataType": data_type}})
                if not isinstance(ds_list, list):
                    ds_list = [ds_list]
                for ds in ds_list:
                    var_key = list(ds.data_vars)[0]
                    da = ds[var_key].squeeze()
                    dim_names = list(da.dims)
                    if any(d in dim_names for d in ['number', 'realization']):
                        m_dim = 'number' if 'number' in dim_names else 'realization'
                        for m_idx in range(len(da[m_dim])):
                            g = standardize_grid(da.isel({m_dim: m_idx}).to_dataset(name=var_key))
                            if g is not None and g.ndim == 2:
                                grids.append(g)
                    else:
                        g = standardize_grid(da.to_dataset(name=var_key))
                        if g is not None and g.ndim == 2:
                            grids.append(g)
            except Exception:
                continue
    except Exception:
        pass
    return grids

def fetch_deterministic(model_name, product, search_str, fxx):
    try:
        H = Herbie(global_init_time, model=model_name, product=product, fxx=fxx)
        ds = H.xarray(search_str)
        if isinstance(ds, list):
            ds = ds[0]
        g = standardize_grid(ds)
        if g is not None and g.ndim == 2:
            return g
    except Exception:
        pass
    return None

def fetch_aigfs(fxx):
    for sq in [":HGT:500 mb:", ":HGT:500:"]:
        try:
            H = Herbie(global_init_time, model="aigfs", product="pres", fxx=fxx)
            ds = H.xarray(sq)
            if isinstance(ds, list):
                ds = ds[0]
            g = standardize_grid(ds)
            if g is not None and g.ndim == 2:
                return g
        except Exception:
            pass
    return None


# --- 5. BATCH PRODUCTION LOOP ---
for fxx in FXX_STEPS:
    valid_time = datetime.strptime(global_init_time, '%Y-%m-%d %H:%M') + timedelta(hours=fxx)
    print(f"\n📦 Processing Lead Time F{fxx:03d} (Valid: {valid_time.strftime('%Y-%m-%d %H:%MZ')})...")
    
    gefs_mems = fetch_gefs_lead(fxx)
    ifs_mems  = fetch_ifs_ensemble(fxx)
    aifs_mems = fetch_aifs_ensemble(fxx)
    
    gefs_mean = np.mean(np.stack(gefs_mems), axis=0) if gefs_mems else None
    ifs_mean  = np.mean(np.stack(ifs_mems), axis=0) if ifs_mems else None
    aifs_mean = np.mean(np.stack(aifs_mems), axis=0) if aifs_mems else None
    all_members = gefs_mems + ifs_mems + aifs_mems
    super_mean = np.mean(np.stack(all_members), axis=0) if all_members else None
    
    det_gfs   = fetch_deterministic("gfs", "pgrb2.0p25", ":HGT:500 mb:", fxx)
    det_ifs   = fetch_deterministic("ifs", "oper", ":gh:500", fxx)
    det_aifs  = fetch_deterministic("aifs", "oper", ":z:500:", fxx)
    det_aigfs = fetch_aigfs(fxx)
    
    super_contours = extract_contours(super_mean)
    print(f"  ↳ Extracted {len(super_contours)} height levels for Superensemble Mean.")

    step_data = {
        "fxx": fxx,
        "init": global_init_time,
        "valid": valid_time.strftime('%Y-%m-%d %H:%MZ'),
        "super_mean": super_contours,
        "gefs_mean": extract_contours(gefs_mean),
        "ifs_mean": extract_contours(ifs_mean),
        "aifs_mean": extract_contours(aifs_mean),
        "gfs_det": extract_contours(det_gfs),
        "ecmwf_det": extract_contours(det_ifs),
        "aifs_det": extract_contours(det_aifs),
        "aigfs_det": extract_contours(det_aigfs),
        "gefs_members": [extract_contours(m) for m in gefs_mems],
        "ifs_members": [extract_contours(m) for m in ifs_mems],
        "aifs_members": [extract_contours(m) for m in aifs_mems]
    }
    
    json_path = os.path.join(DATA_DIR, f"f{fxx:03d}.json")
    with open(json_path, "w") as f:
        json.dump(step_data, f, separators=(',', ':'))

plt.close(fig)


# --- 6. WRITE INTERACTIVE HTML/JS FRONTEND ---
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>500mb Multi-Ensemble Interactive Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {{
      --bg: #0b1120;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #3b82f6;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0; background: var(--bg); color: var(--text); display: flex; flex-direction: column; align-items: center;
    }}
    .header {{
      width: 100%; max-width: 1250px; padding: 16px 20px 8px 20px; box-sizing: border-box; text-align: center;
    }}
    .header h2 {{ margin: 0 0 6px 0; font-size: 1.4rem; }}
    .header .subtitle {{ font-size: 0.9rem; color: var(--text-muted); }}

    .toolbar {{
      width: 100%; max-width: 1250px; background: var(--card-bg); border: 1px solid var(--border);
      border-radius: 10px; padding: 12px 16px; margin: 10px 0; box-sizing: border-box;
      display: flex; flex-direction: column; gap: 12px;
    }}
    .row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 12px; justify-content: space-between; }}
    .btn {{
      background: var(--accent); color: white; border: none; padding: 6px 14px; border-radius: 6px;
      cursor: pointer; font-size: 13px; font-weight: 600; transition: 0.15s;
    }}
    .btn:hover {{ filter: brightness(1.15); }}
    .btn-secondary {{ background: #475569; }}
    select {{
      padding: 6px 10px; border-radius: 6px; background: #0f172a; color: white;
      border: 1px solid var(--border); font-size: 13px; cursor: pointer;
    }}
    
    .checkbox-group {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center; font-size: 13px; }}
    .checkbox-item {{ display: flex; align-items: center; gap: 6px; cursor: pointer; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 2px; display: inline-block; }}
    
    .contour-section {{ border-top: 1px solid var(--border); padding-top: 10px; display: flex; flex-direction: column; gap: 8px; }}
    .contour-chips {{ display: flex; flex-wrap: wrap; gap: 5px; max-height: 80px; overflow-y: auto; }}
    .chip {{
      background: #0f172a; border: 1px solid var(--border); color: #cbd5e1;
      padding: 3px 8px; border-radius: 4px; font-size: 12px; cursor: pointer; user-select: none;
    }}
    .chip.active {{ background: #2563eb; color: white; border-color: #3b82f6; font-weight: bold; }}

    .viewer-card {{
      position: relative; width: 100%; max-width: 1200px; aspect-ratio: 1200 / 800;
      background: #000; border-radius: 10px; overflow: hidden; box-shadow: 0 12px 30px rgba(0,0,0,0.5);
    }}
    .viewer-card img, .viewer-card canvas {{
      position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    }}
    #contourCanvas {{ pointer-events: none; }}
    
    .legend-box {{
      position: absolute; top: 12px; right: 12px; background: rgba(255, 255, 255, 0.92);
      color: #111; padding: 8px 12px; border-radius: 6px; font-size: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.25); display: flex; flex-direction: column; gap: 4px;
      pointer-events: none; max-width: 220px;
    }}
    .legend-line {{ display: flex; align-items: center; gap: 8px; }}
    .line-sample {{ width: 22px; height: 3px; display: inline-block; }}

    .title-stamp {{
      position: absolute; top: 12px; left: 12px; background: rgba(255, 255, 255, 0.92);
      color: #111; padding: 6px 10px; border-radius: 6px; font-size: 12px; font-weight: bold;
      pointer-events: none; line-height: 1.3;
    }}
    .footer {{ margin: 15px 0 30px 0; font-size: 12px; color: var(--text-muted); }}
    kbd {{ background: #334155; padding: 2px 5px; border-radius: 4px; font-size: 11px; }}
  </style>
</head>
<body>

  <div class="header">
    <h2>500 hPa Multi-Ensemble & Deterministic Spaghetti Viewer</h2>
    <div class="subtitle">Baseline Cycle: {global_init_time}Z | Full 240-Hour Range</div>
  </div>

  <div class="toolbar">
    <!-- Row 1: Lead Scrubbing & View Mode -->
    <div class="row">
      <div style="display: flex; gap: 8px; align-items: center;">
        <button class="btn btn-secondary" onclick="step(-1)">◀ Prev</button>
        <select id="leadSelect" onchange="loadStepData()">
          {"".join([f'<option value="{f:03d}">F{f:03d} (+{f}h)</option>' for f in FXX_STEPS])}
        </select>
        <button class="btn btn-secondary" onclick="step(1)">Next ▶</button>
      </div>

      <div style="display: flex; gap: 14px; align-items: center;">
        <b>Display Mode:</b>
        <label><input type="radio" name="displayMode" value="spaghetti" checked onchange="draw()"> All Members (Spaghetti) + Means</label>
        <label><input type="radio" name="displayMode" value="means" onchange="draw()"> Means Only</label>
      </div>
    </div>

    <!-- Row 2: Model & Ensemble Toggles -->
    <div class="checkbox-group">
      <b>Models:</b>
      <label class="checkbox-item"><input type="checkbox" id="chk_super" checked onchange="draw()"><span class="swatch" style="background:#000;"></span> Super Mean</label>
      <label class="checkbox-item"><input type="checkbox" id="chk_gefs" checked onchange="draw()"><span class="swatch" style="background:#0033cc;"></span> GEFS</label>
      <label class="checkbox-item"><input type="checkbox" id="chk_ifs" checked onchange="draw()"><span class="swatch" style="background:#c0392b;"></span> ECMWF IFS</label>
      <label class="checkbox-item"><input type="checkbox" id="chk_aifs" checked onchange="draw()"><span class="swatch" style="background:#196f3d;"></span> AIFS</label>
      <label class="checkbox-item"><input type="checkbox" id="chk_gfs" checked onchange="draw()"><span class="swatch" style="background:#001155; border-bottom: 2px dashed #fff;"></span> GFS Det</label>
      <label class="checkbox-item"><input type="checkbox" id="chk_aigfs" checked onchange="draw()"><span class="swatch" style="background:#8e44ad; border-bottom: 2px dashed #fff;"></span> AIGFS Det</label>
      <label class="checkbox-item"><input type="checkbox" id="chk_ecmwf_det" onchange="draw()"><span class="swatch" style="background:#990000; border-bottom: 2px dashed #fff;"></span> IFS Det</label>
      <label class="checkbox-item"><input type="checkbox" id="chk_aifs_det" onchange="draw()"><span class="swatch" style="background:#009933; border-bottom: 2px dashed #fff;"></span> AIFS Det</label>
    </div>

    <!-- Row 3: Height Contours (4980 to 6000m) -->
    <div class="contour-section">
      <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
        <span style="font-size: 13px; font-weight: bold;">Contour Levels (m):</span>
        <button class="btn btn-secondary" style="padding: 2px 8px; font-size: 11px;" onclick="selectAllContours(true)">Select All</button>
        <button class="btn btn-secondary" style="padding: 2px 8px; font-size: 11px;" onclick="setStandardContours()">Standard 500mb (5400-5880)</button>
        <button class="btn btn-secondary" style="padding: 2px 8px; font-size: 11px;" onclick="selectAllContours(false)">Clear All</button>
      </div>
      <div class="contour-chips" id="contourChips"></div>
    </div>
  </div>

  <!-- Map Card -->
  <div class="viewer-card">
    <img src="basemap.png" alt="CONUS Basemap">
    <canvas id="contourCanvas" width="1200" height="800"></canvas>
    
    <div class="title-stamp" id="mapTitleStamp">
      500 hPa Height Contours<br>Loading...
    </div>

    <div class="legend-box" id="legendBox"></div>
  </div>

  <div class="footer">
    Shortcuts: Use <kbd>←</kbd> and <kbd>→</kbd> arrow keys to step forecast lead times.
  </div>

  <script>
    const ALL_CONTOURS = {ALL_CONTOURS};
    const STANDARD_CONTOURS = [5400, 5520, 5640, 5760, 5880];
    let selectedContours = new Set(ALL_CONTOURS);
    let dataCache = {{}};
    let currentStepData = null;

    const chipContainer = document.getElementById('contourChips');
    ALL_CONTOURS.forEach(lvl => {{
      const chip = document.createElement('div');
      chip.className = 'chip active';
      chip.id = `chip_${{lvl}}`;
      chip.textContent = lvl;
      chip.onclick = () => toggleContour(lvl);
      chipContainer.appendChild(chip);
    }});

    function toggleContour(lvl) {{
      const chip = document.getElementById(`chip_${{lvl}}`);
      if (selectedContours.has(lvl)) {{
        selectedContours.delete(lvl);
        chip.classList.remove('active');
      }} else {{
        selectedContours.add(lvl);
        chip.classList.add('active');
      }}
      draw();
    }}

    function selectAllContours(enable) {{
      ALL_CONTOURS.forEach(lvl => {{
        const chip = document.getElementById(`chip_${{lvl}}`);
        if (enable) {{
          selectedContours.add(lvl);
          chip.classList.add('active');
        }} else {{
          selectedContours.delete(lvl);
          chip.classList.remove('active');
        }}
      }});
      draw();
    }}

    function setStandardContours() {{
      selectAllContours(false);
      STANDARD_CONTOURS.forEach(lvl => {{
        selectedContours.add(lvl);
        const chip = document.getElementById(`chip_${{lvl}}`);
        if (chip) chip.classList.add('active');
      }});
      draw();
    }}

    async function loadStepData() {{
      const fxx = document.getElementById('leadSelect').value;
      if (dataCache[fxx]) {{
        currentStepData = dataCache[fxx];
        draw();
        return;
      }}
      try {{
        const resp = await fetch(`data/f${{fxx}}.json`);
        const json = await resp.json();
        dataCache[fxx] = json;
        currentStepData = json;
        draw();
      }} catch (err) {{
        console.error("Error loading JSON for lead F" + fxx, err);
      }}
    }}

    function step(delta) {{
      const select = document.getElementById('leadSelect');
      let idx = select.selectedIndex + delta;
      if (idx >= 0 && idx < select.options.length) {{
        select.selectedIndex = idx;
        loadStepData();
      }}
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowLeft') step(-1);
      if (e.key === 'ArrowRight') step(1);
    }});

    function draw() {{
      if (!currentStepData) return;
      const canvas = document.getElementById('contourCanvas');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const isSpaghetti = document.querySelector('input[name="displayMode"]:checked').value === 'spaghetti';
      const showSuper = document.getElementById('chk_super').checked;
      const showGefs = document.getElementById('chk_gefs').checked;
      const showIfs = document.getElementById('chk_ifs').checked;
      const showAifs = document.getElementById('chk_aifs').checked;
      const showGfs = document.getElementById('chk_gfs').checked;
      const showAigfs = document.getElementById('chk_aigfs').checked;
      const showEcmwf = document.getElementById('chk_ecmwf_det').checked;
      const showAifsDet = document.getElementById('chk_aifs_det').checked;

      document.getElementById('mapTitleStamp').innerHTML = 
        `500 hPa Height (m) | Lead: F${{String(currentStepData.fxx).padStart(3, '0')}}<br>` +
        `Init: ${{currentStepData.init}}Z | Valid: ${{currentStepData.valid}}`;

      const legendItems = [];

      function drawLines(layerDict, color, width, alpha, dashed = false) {{
        if (!layerDict) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.globalAlpha = alpha;
        ctx.setLineDash(dashed ? [6, 4] : []);

        selectedContours.forEach(lvl => {{
          const lines = layerDict[String(lvl)];
          if (lines) {{
            lines.forEach(pts => {{
              if (pts.length < 2) return;
              ctx.beginPath();
              ctx.moveTo(pts[0][0], pts[0][1]);
              for (let i = 1; i < pts.length; i++) {{
                ctx.lineTo(pts[i][0], pts[i][1]);
              }}
              ctx.stroke();
            }});
          }}
        }});
      }}

      // 1. Members
      if (isSpaghetti) {{
        if (showGefs && currentStepData.gefs_members) {{
          currentStepData.gefs_members.forEach(m => drawLines(m, '#5b9bd5', 0.6, 0.35));
          legendItems.push({{ label: `GEFS Members (${{currentStepData.gefs_members.length}})`, color: '#5b9bd5', dash: false }});
        }}
        if (showIfs && currentStepData.ifs_members) {{
          currentStepData.ifs_members.forEach(m => drawLines(m, '#f1948a', 0.6, 0.35));
          legendItems.push({{ label: `IFS Members (${{currentStepData.ifs_members.length}})`, color: '#f1948a', dash: false }});
        }}
        if (showAifs && currentStepData.aifs_members) {{
          currentStepData.aifs_members.forEach(m => drawLines(m, '#82e0aa', 0.6, 0.35));
          legendItems.push({{ label: `AIFS Members (${{currentStepData.aifs_members.length}})`, color: '#82e0aa', dash: false }});
        }}
      }}

      // 2. Deterministics
      if (showGfs && currentStepData.gfs_det) {{
        drawLines(currentStepData.gfs_det, '#001155', 1.8, 1.0, true);
        legendItems.push({{ label: 'GFS Det', color: '#001155', dash: true }});
      }}
      if (showAigfs && currentStepData.aigfs_det) {{
        drawLines(currentStepData.aigfs_det, '#8e44ad', 1.8, 1.0, true);
        legendItems.push({{ label: 'AIGFS Det', color: '#8e44ad', dash: true }});
      }}
      if (showEcmwf && currentStepData.ecmwf_det) {{
        drawLines(currentStepData.ecmwf_det, '#990000', 1.8, 1.0, true);
        legendItems.push({{ label: 'IFS Det', color: '#990000', dash: true }});
      }}
      if (showAifsDet && currentStepData.aifs_det) {{
        drawLines(currentStepData.aifs_det, '#009933', 1.8, 1.0, true);
        legendItems.push({{ label: 'AIFS Det', color: '#009933', dash: true }});
      }}

      // 3. Means
      if (showGefs && currentStepData.gefs_mean) {{
        drawLines(currentStepData.gefs_mean, '#0033cc', 2.2, 1.0);
        legendItems.push({{ label: 'GEFS Mean', color: '#0033cc', dash: false }});
      }}
      if (showIfs && currentStepData.ifs_mean) {{
        drawLines(currentStepData.ifs_mean, '#c0392b', 2.2, 1.0);
        legendItems.push({{ label: 'IFS Mean', color: '#c0392b', dash: false }});
      }}
      if (showAifs && currentStepData.aifs_mean) {{
        drawLines(currentStepData.aifs_mean, '#196f3d', 2.2, 1.0);
        legendItems.push({{ label: 'AIFS Mean', color: '#196f3d', dash: false }});
      }}

      // 4. Superensemble Mean
      if (showSuper && currentStepData.super_mean) {{
        drawLines(currentStepData.super_mean, '#000000', 3.0, 1.0);
        legendItems.push({{ label: 'Superensemble Mean', color: '#000000', dash: false }});

        ctx.font = 'bold 11px sans-serif';
        ctx.fillStyle = '#000';
        ctx.globalAlpha = 1.0;
        selectedContours.forEach(lvl => {{
          const lines = currentStepData.super_mean[String(lvl)];
          if (lines) {{
            lines.forEach(pts => {{
              if (pts.length > 10) {{
                const mid = pts[Math.floor(pts.length / 2)];
                ctx.fillText(`${{lvl}} m`, mid[0] + 4, mid[1] - 4);
              }}
            }});
          }}
        }});
      }}

      const legendBox = document.getElementById('legendBox');
      legendBox.innerHTML = legendItems.map(item => `
        <div class="legend-line">
          <span class="line-sample" style="background:${{item.color}}; border-bottom: ${{item.dash ? '2px dashed #000' : 'none'}};"></span>
          <span>${{item.label}}</span>
        </div>
      `).join('');
    }}

    loadStepData();
  </script>
</body>
</html>
"""

with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
    f.write(html_content)

print(f"🎉 Production build complete! Files written to {OUTPUT_DIR}")
