import os
import json
import warnings
import concurrent.futures
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Headless backend for automation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DIRECTORIES & RUN CONFIGURATION
# ==============================================================================
OUTPUT_DIR = "output"
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
SPREAD_DIR = os.path.join(OUTPUT_DIR, "spread")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SPREAD_DIR, exist_ok=True)

# 0 to 240 hours every 12 hours (21 forecast steps)
FXX_STEPS = list(range(0, 241, 12))
ALL_CONTOURS = list(range(4980, 6001, 60))

# Fixed canvas projection dimensions (1200x800)
CANVAS_W, CANVAS_H = 1200, 800

# Operational 00Z / 12Z baseline cycle detector
now = datetime.utcnow()
if (now.hour == 1 and now.minute > 30) or (2 <= now.hour <= 12) or (now.hour == 13 and now.minute <= 30):
    base_date = now - timedelta(days=1)
    global_cycle = "12"
else:
    base_date = now - timedelta(days=1) if now.hour <= 1 else now
    global_cycle = "00"

global_init_time = base_date.strftime(f'%Y-%m-%d {global_cycle}:00')
print(f"📡 Global Model Baseline Cycle: {global_init_time}")

# Synoptic CONUS domain
lon_min, lon_max = -130.0, -60.0
lat_min, lat_max = 20.0, 55.0
grid_lats = np.arange(lat_min, lat_max + 0.5, 0.5)
grid_lons = np.arange(lon_min, lon_max + 0.5, 0.5)
target_lons, target_lats = np.meshgrid(grid_lons, grid_lats)


# ==============================================================================
# 2. BASEMAP GENERATION & GEOGRAPHIC TRANSFORM
# ==============================================================================
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

# Transform mapping PlateCarree (lon, lat) -> Screen Pixel Coordinates
geo_trans = ccrs.PlateCarree()._as_mpl_transform(ax)


# ==============================================================================
# 3. GRID STANDARDIZATION & VECTOR EXTRACTION
# ==============================================================================
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
        # Unit conversion: if geopotential (m^2/s^2 ~ 50,000+), convert to gpm
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
                disp_coords = geo_trans.transform(s)
                fig_coords = fig.transFigure.inverted().transform(disp_coords)
                
                px = np.round(fig_coords[:, 0] * CANVAS_W).astype(int)
                py = np.round((1.0 - fig_coords[:, 1]) * CANVAS_H).astype(int)
                
                pts = []
                for x, y in zip(px, py):
                    if -50 <= x <= CANVAS_W + 50 and -50 <= y <= CANVAS_H + 50:
                        if not pts or (abs(x - pts[-1][0]) >= 2 or abs(y - pts[-1][1]) >= 2):
                            pts.append([int(x), int(y)])
                if len(pts) > 1:
                    lvl_lines.append(pts)
            if lvl_lines:
                result[lvl_key] = lvl_lines

        if hasattr(cs, 'remove'):
            cs.remove()
        elif hasattr(cs, 'collections'):
            for coll in cs.collections:
                coll.remove()
        return result
    except Exception:
        return {}


# ==============================================================================
# 4. DATA FETCHERS (BALANCED 2x2 ENSEMBLE + DETERMINISTICS)
# ==============================================================================
# 1. GEFS (Physics - NCEP)
def fetch_gefs_lead(fxx):
    members = ["c00"] + [f"p{str(i).zfill(2)}" for i in range(1, 31)]
    def _worker(m):
        try:
            H = Herbie(global_init_time, model="gefs", product="atmos.5", member=m, fxx=fxx)
            return standardize_grid(H.xarray(":HGT:500 mb:"))
        except Exception:
            return None
    grids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(_worker, members):
            if res is not None and res.ndim == 2:
                grids.append(res)
    return grids

# 2. AIGEFS (AI Emulator - NCEP)
def fetch_aigefs_ensemble(fxx):
    members = ["c00"] + [f"p{str(i).zfill(2)}" for i in range(1, 31)]
    def _worker(m):
        for sq in [":HGT:500 mb:", ":HGT:500:"]:
            try:
                H = Herbie(global_init_time, model="aigefs", product="pres", member=m, fxx=fxx)
                ds = H.xarray(sq)
                return standardize_grid(ds)
            except Exception:
                continue
        return None
    grids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(_worker, members):
            if res is not None and res.ndim == 2:
                grids.append(res)
                
    # Fallback to ensemble mean if individual members are pending
    if not grids:
        try:
            H_mean = Herbie(global_init_time, model="aigefs", product="pres", member="avg", fxx=fxx)
            ds = H_mean.xarray(":HGT:500 mb:")
            g = standardize_grid(ds)
            if g is not None:
                grids.append(g)
        except Exception:
            pass
    return grids

# 3. ECMWF IFS ENS (Physics - ECMWF)
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

# 4. AIFS ENS (AI Emulator - ECMWF)
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

# Deterministic Fetchers
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


# ==============================================================================
# 5. SPREAD & DISAGREEMENT UNDERLAY ENGINE
# ==============================================================================
SPREAD_LEVELS = np.arange(10, 130, 10)  # 10m to 120m spread intervals
SPREAD_CMAP = plt.cm.YlOrRd

def compute_spread_diagnostics(gefs_mems, aigefs_mems, ifs_mems, aifs_mems):
    """Calculates internal intra-spreads, pooled intra-spread, and inter-model disagreement."""
    def calc_spread(mem_list):
        if len(mem_list) > 1:
            return np.std(np.stack(mem_list), axis=0, ddof=1)
        return None

    spr_gefs = calc_spread(gefs_mems)
    spr_aigefs = calc_spread(aigefs_mems)
    spr_ifs = calc_spread(ifs_mems)
    spr_aifs = calc_spread(aifs_mems)

    # Calculate system means and collect internal variances
    means = []
    intra_vars = []
    for mems, spr in [(gefs_mems, spr_gefs), (aigefs_mems, spr_aigefs), 
                      (ifs_mems, spr_ifs), (aifs_mems, spr_aifs)]:
        if mems:
            means.append(np.mean(np.stack(mems), axis=0))
        if spr is not None:
            intra_vars.append(spr**2)

    # Inter-Model Disagreement (Variance among active ensemble means)
    if len(means) > 1:
        spr_inter = np.std(np.stack(means), axis=0, ddof=1)
    else:
        spr_inter = np.zeros_like(target_lats)

    # Pooled Internal Chaos
    if intra_vars:
        spr_intra_pooled = np.sqrt(np.mean(np.stack(intra_vars), axis=0))
    else:
        spr_intra_pooled = np.zeros_like(target_lats)

    # Total Superensemble Spread
    all_members = gefs_mems + aigefs_mems + ifs_mems + aifs_mems
    if len(all_members) > 1:
        spr_total = np.std(np.stack(all_members), axis=0, ddof=1)
    else:
        spr_total = np.zeros_like(target_lats)

    return {
        "total": spr_total,
        "inter": spr_inter,
        "intra_pooled": spr_intra_pooled,
        "gefs": spr_gefs,
        "aigefs": spr_aigefs,
        "ifs": spr_ifs,
        "aifs": spr_aifs
    }

def render_spread_underlay(spread_grid, filename):
    """Renders shaded background contours matching basemap dimensions."""
    if spread_grid is None or np.all(np.isnan(spread_grid)):
        return
    fig_spr = plt.figure(figsize=(12, 8), dpi=100)
    ax_spr = fig_spr.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0))
    ax_spr.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax_spr.add_feature(cfeature.LAND, facecolor="#f8f9fa")
    ax_spr.add_feature(cfeature.OCEAN, facecolor="#eef2f5")
    
    # Render shaded contours
    ax_spr.contourf(
        target_lons, target_lats, spread_grid,
        levels=SPREAD_LEVELS, cmap=SPREAD_CMAP,
        transform=ccrs.PlateCarree(), extend='max', alpha=0.85
    )
    
    ax_spr.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#333333")
    ax_spr.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="#333333")
    ax_spr.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#666666", linestyle=":")
    ax_spr.add_feature(cfeature.LAKES, facecolor="#eef2f5", edgecolor="#666666", linewidth=0.4)

    fig_spr.savefig(os.path.join(SPREAD_DIR, filename), dpi=100)
    plt.close(fig_spr)


# ==============================================================================
# 6. BATCH PRODUCTION LOOP
# ==============================================================================
for fxx in FXX_STEPS:
    valid_time = datetime.strptime(global_init_time, '%Y-%m-%d %H:%M') + timedelta(hours=fxx)
    print(f"\n📦 Processing Lead Time F{fxx:03d} (Valid: {valid_time.strftime('%Y-%m-%d %H:%MZ')})...")
    
    # 1. Fetch all 4 ensembles
    gefs_mems   = fetch_gefs_lead(fxx)
    aigefs_mems = fetch_aigefs_ensemble(fxx)
    ifs_mems    = fetch_ifs_ensemble(fxx)
    aifs_mems   = fetch_aifs_ensemble(fxx)

    print(f"  ↳ Members loaded: GEFS={len(gefs_mems)}, AIGEFS={len(aigefs_mems)}, IFS={len(ifs_mems)}, AIFS={len(aifs_mems)}")

    # 2. Compute means
    gefs_mean   = np.mean(np.stack(gefs_mems), axis=0) if gefs_mems else None
    aigefs_mean = np.mean(np.stack(aigefs_mems), axis=0) if aigefs_mems else None
    ifs_mean    = np.mean(np.stack(ifs_mems), axis=0) if ifs_mems else None
    aifs_mean   = np.mean(np.stack(aifs_mems), axis=0) if aifs_mems else None
    
    all_members = gefs_mems + aigefs_mems + ifs_mems + aifs_mems
    super_mean  = np.mean(np.stack(all_members), axis=0) if all_members else None

    # 3. Compute and render Spread Underlays
    spread_diags = compute_spread_diagnostics(gefs_mems, aigefs_mems, ifs_mems, aifs_mems)
    for spr_key, spr_grid in spread_diags.items():
        if spr_grid is not None:
            render_spread_underlay(spr_grid, f"{spr_key}_f{fxx:03d}.png")

    # 4. Fetch deterministics
    det_gfs   = fetch_deterministic("gfs", "pgrb2.0p25", ":HGT:500 mb:", fxx)
    det_ifs   = fetch_deterministic("ifs", "oper", ":gh:500", fxx)
    det_aifs  = fetch_deterministic("aifs", "oper", ":z:500:", fxx)
    det_aigfs = fetch_aigfs(fxx)

    # 5. Extract vector contours
    super_contours = extract_contours(super_mean)
    print(f"  ↳ Extracted {len(super_contours)} height levels for Superensemble Mean.")

    step_data = {
        "fxx": fxx,
        "init": global_init_time,
        "valid": valid_time.strftime('%Y-%m-%d %H:%MZ'),
        "super_mean": super_contours,
        "gefs_mean": extract_contours(gefs_mean),
        "aigefs_mean": extract_contours(aigefs_mean),
        "ifs_mean": extract_contours(ifs_mean),
        "aifs_mean": extract_contours(aifs_mean),
        "gfs_det": extract_contours(det_gfs),
        "aigfs_det": extract_contours(det_aigfs),
        "ecmwf_det": extract_contours(det_ifs),
        "aifs_det": extract_contours(det_aifs),
        "gefs_members": [extract_contours(m) for m in gefs_mems],
        "aigefs_members": [extract_contours(m) for m in aigefs_mems],
        "ifs_members": [extract_contours(m) for m in ifs_mems],
        "aifs_members": [extract_contours(m) for m in aifs_mems]
    }

    json_path = os.path.join(DATA_DIR, f"f{fxx:03d}.json")
    with open(json_path, "w") as f:
        json.dump(step_data, f, separators=(',', ':'))

plt.close(fig)


# ==============================================================================
# 7. GENERATE DYNAMIC HTML FRONTEND
# ==============================================================================
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
      position: absolute; top: 12px; right: 12px; background: rgba(255, 255, 255, 0.94);
      color: #111; padding: 8px 12px; border-radius: 6px; font-size: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.25); display: flex; flex-direction: column; gap: 4px;
      pointer-events: none; max-width: 240px;
    }}
    .legend-line {{ display: flex; align-items: center; gap: 8px; }}
    .line-sample {{ width: 22px; height: 3px; display: inline-block; }}

    .title-stamp {{
      position: absolute; top: 12px; left: 12px; background: rgba(255, 255, 255, 0.94);
      color: #111; padding: 6px 10px; border-radius: 6px; font-size: 12px; font-weight: bold;
      pointer-events: none; line-height: 1.3;
    }}
    
    /* Spread Colorbar Strip */
    .spread-colorbar {{
      display: none; align-items: center; gap: 8px; font-size: 11px;
    }}
    .colorbar-gradient {{
      width: 140px; height: 12px; border-radius: 3px;
      background: linear-gradient(to right, #ffffcc, #fed976, #feb24c, #fd8d3c, #f03b20, #bd0026);
      border: 1px solid var(--border);
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
    <!-- Row 1: Forecast Lead, Display Mode, and Background Shading -->
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
        <label><input type="radio" name="displayMode" value="spaghetti" checked onchange="draw()"> Spaghetti + Means</label>
        <label><input type="radio" name="displayMode" value="means" onchange="draw()"> Means Only</label>
      </div>

      <div style="display: flex; gap: 8px; align-items: center;">
        <b>Underlay:</b>
        <select id="bgSelect" onchange="updateBackground()">
          <option value="none">Basemap Only</option>
          <option value="inter">Inter-Model Disagreement (σ_inter)</option>
          <option value="total">Total Super Spread (σ_total)</option>
          <option value="intra_pooled">Pooled Internal Chaos (σ_intra)</option>
          <option value="gefs">GEFS Spread</option>
          <option value="aigefs">AIGEFS Spread</option>
          <option value="ifs">ECMWF IFS Spread</option>
          <option value="aifs">AIFS Spread</option>
        </select>

        <div class="spread-colorbar" id="cbarBox">
          <span>10m</span>
          <div class="colorbar-gradient"></div>
          <span>120m+</span>
        </div>
      </div>
    </div>

    <!-- Row 2: Ensemble & Deterministic Toggles -->
    <div class="checkbox-group">
      <b>Models:</b>
      <label class="checkbox-item"><input type="checkbox" id="chk_super" checked onchange="draw()"><span class="swatch" style="background:#000;"></span> Super Mean</label>
      <label class="checkbox-item"><input type="checkbox" id="chk_gefs" checked onchange="draw()"><span class="swatch" style="background:#0033cc;"></span> GEFS</label>
      <label class="checkbox-item"><input type="checkbox" id="chk_aigefs" checked onchange="draw()"><span class="swatch" style="background:#6c3483;"></span> AIGEFS</label>
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
    <img id="mapUnderlay" src="basemap.png" alt="CONUS Basemap">
    <canvas id="contourCanvas" width="1200" height="800"></canvas>
    
    <div class="title-stamp" id="mapTitleStamp">
      500 hPa Height Contours<br>Loading...
    </div>

    <div class="legend-box" id="legendBox"></div>
  </div>

  <div class="footer">
    Shortcuts: Use <kbd>←</kbd> and <kbd>→</kbd> arrow keys to scrub forecast lead times.
  </div>

  <script>
    const ALL_CONTOURS = {ALL_CONTOURS};
    const STANDARD_CONTOURS = [5400, 5520, 5640, 5760, 5880];
    let selectedContours = new Set(ALL_CONTOURS);
    let dataCache = {{}};
    let currentStepData = null;

    // Contour Selector Chips
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

    // Underlay Management
    function updateBackground() {{
      const bg = document.getElementById('bgSelect').value;
      const fxx = document.getElementById('leadSelect').value;
      const img = document.getElementById('mapUnderlay');
      const cbar = document.getElementById('cbarBox');

      if (bg === 'none') {{
        img.src = 'basemap.png';
        cbar.style.display = 'none';
      }} else {{
        img.src = `spread/${{bg}}_f${{fxx}}.png`;
        cbar.style.display = 'flex';
      }}
    }}

    // Load Step Data
    async function loadStepData() {{
      const fxx = document.getElementById('leadSelect').value;
      updateBackground();

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

    // Canvas Renderer
    function draw() {{
      if (!currentStepData) return;
      const canvas = document.getElementById('contourCanvas');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const isSpaghetti = document.querySelector('input[name="displayMode"]:checked').value === 'spaghetti';
      const showSuper = document.getElementById('chk_super').checked;
      const showGefs = document.getElementById('chk_gefs').checked;
      const showAigefs = document.getElementById('chk_aigefs').checked;
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

      // 1. Members (Spaghetti)
      if (isSpaghetti) {{
        if (showGefs && currentStepData.gefs_members) {{
          currentStepData.gefs_members.forEach(m => drawLines(m, '#5b9bd5', 0.6, 0.35));
          legendItems.push({{ label: `GEFS Members (${{currentStepData.gefs_members.length}})`, color: '#5b9bd5', dash: false }});
        }}
        if (showAigefs && currentStepData.aigefs_members) {{
          currentStepData.aigefs_members.forEach(m => drawLines(m, '#bb8fce', 0.6, 0.35));
          legendItems.push({{ label: `AIGEFS Members (${{currentStepData.aigefs_members.length}})`, color: '#bb8fce', dash: false }});
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

      // 2. Deterministics (Dashed)
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

      // 3. Ensemble Means
      if (showGefs && currentStepData.gefs_mean) {{
        drawLines(currentStepData.gefs_mean, '#0033cc', 2.2, 1.0);
        legendItems.push({{ label: 'GEFS Mean', color: '#0033cc', dash: false }});
      }}
      if (showAigefs && currentStepData.aigefs_mean) {{
        drawLines(currentStepData.aigefs_mean, '#6c3483', 2.2, 1.0);
        legendItems.push({{ label: 'AIGEFS Mean', color: '#6c3483', dash: false }});
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

      // Dynamic Legend
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
