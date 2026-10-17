import os
import json
import shutil
import warnings
import tempfile
import concurrent.futures
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from herbie import Herbie

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DIRECTORIES & RUN CONFIGURATION
# ==============================================================================
OUTPUT_DIR = "output"
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# 0 to 240 hours every 12 hours (21 lead times)
FXX_STEPS = list(range(0, 241, 12))
ALL_CONTOURS = list(range(4980, 6001, 60))

# Fixed canvas dimensions (1200x800)
CANVAS_W, CANVAS_H = 1200, 800

# High-performance HTTP Session with connection pooling and retries
http_session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
http_session.mount("https://", adapter)
http_session.mount("http://", adapter)
http_session.headers.update({"User-Agent": "Mozilla/5.0 (500mb-ensemble-viewer; Operational Met Research)"})


# ==============================================================================
# 2. DYNAMIC CYCLE DETECTOR (ECMWF IFS F240 COMPLETION TRIGGER)
# ==============================================================================
def detect_latest_complete_cycle():
    """Probes candidate synoptic cycles and locks onto the latest completed F240 run."""
    now = datetime.utcnow()
    latest_hour = 12 if now.hour >= 12 else 0
    anchor = now.replace(hour=latest_hour, minute=0, second=0, microsecond=0)
    
    candidates = [anchor - timedelta(hours=12 * i) for i in range(4)]
    print("🔍 Probing for newest complete 240h ensemble cycle...")
    
    for cand in candidates:
        cycle_str = cand.strftime('%Y-%m-%d %H:00')
        print(f"  • Checking candidate cycle: {cycle_str}Z...")
        
        # Probe ECMWF IFS ENS at F240
        ifs_ready = False
        try:
            H_ifs = Herbie(cycle_str, model="ifs", product="enfo", fxx=240)
            if H_ifs.grib is not None:
                ifs_ready = True
        except Exception:
            pass

        if not ifs_ready:
            print(f"    ↳ ECMWF IFS F240 not complete for {cycle_str}Z.")
            continue

        # Probe NOAA AIGEFS at F240 (control member: mem000)
        aigefs_ready = False
        date_nomads = cand.strftime('%Y%m%d')
        cycle_nomads = cand.strftime('%H')
        url_aigefs = (
            f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod/"
            f"aigefs.{date_nomads}/{cycle_nomads}/mem000/model/atmos/grib2/"
            f"aigefs.t{cycle_nomads}z.pres.f240.grib2.idx"
        )
        try:
            resp = http_session.head(url_aigefs, timeout=5)
            if resp.status_code == 200:
                aigefs_ready = True
            else:
                H_ai = Herbie(cycle_str, model="aigefs", product="pres", member="mem000", fxx=240)
                if H_ai.grib is not None:
                    aigefs_ready = True
        except Exception:
            pass

        if ifs_ready and aigefs_ready:
            print(f"🎯 Target Cycle Verified: {cycle_str}Z (All global suites complete out to 240h)\n")
            return cycle_str

    fallback_str = candidates[1].strftime('%Y-%m-%d %H:00')
    print(f"⚠️ Falling back to verified baseline: {fallback_str}Z\n")
    return fallback_str

global_init_time = detect_latest_complete_cycle()
date_dt = datetime.strptime(global_init_time, '%Y-%m-%d %H:%M')
current_cycle_id = date_dt.strftime('%Y%m%d_%H')

# Define target directories for the active cycle
CURRENT_RUN_DIR = os.path.join(RUNS_DIR, current_cycle_id)
CURRENT_DATA_DIR = os.path.join(CURRENT_RUN_DIR, "data")
CURRENT_SPREAD_DIR = os.path.join(CURRENT_RUN_DIR, "spread")
os.makedirs(CURRENT_DATA_DIR, exist_ok=True)
os.makedirs(CURRENT_SPREAD_DIR, exist_ok=True)

# Synoptic CONUS domain
lon_min, lon_max = -130.0, -60.0
lat_min, lat_max = 20.0, 55.0
grid_lats = np.arange(lat_min, lat_max + 0.5, 0.5)
grid_lons = np.arange(lon_min, lon_max + 0.5, 0.5)
target_lons, target_lats = np.meshgrid(grid_lons, grid_lats)


# ==============================================================================
# 3. BASEMAP GENERATION & GEOGRAPHIC TRANSFORM
# ==============================================================================
basemap_path = os.path.join(OUTPUT_DIR, "basemap.png")
fig = plt.figure(figsize=(12, 8), dpi=100)
ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0))
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

if not os.path.exists(basemap_path):
    print("🗺️ Rendering static Cartopy basemap...")
    ax.add_feature(cfeature.LAND, facecolor="#f8f9fa")
    ax.add_feature(cfeature.OCEAN, facecolor="#eef2f5")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.1, edgecolor="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor="#333333")
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#888888", linestyle=":")
    ax.add_feature(cfeature.LAKES, facecolor="#eef2f5", edgecolor="#888888", linewidth=0.4)
    fig.savefig(basemap_path, dpi=100)
    print("✅ Saved basemap.png")

geo_trans = ccrs.PlateCarree()._as_mpl_transform(ax)


# ==============================================================================
# 4. GRID STANDARDIZATION & VECTOR EXTRACTION
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
        if np.nanmean(arr) > 20000:
            arr = arr / 9.80665
        return arr
    except Exception:
        return None

def extract_contours(grid):
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
# 5. DATA FETCHERS (BALANCED 2x2 ENSEMBLES + DETERMINISTICS)
# ==============================================================================
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

def fetch_aigefs_direct_http(date_str, cycle_str, member_str, fxx):
    fxx_str = f"f{fxx:03d}"
    url_base = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod/aigefs.{date_str}/{cycle_str}/{member_str}/model/atmos/grib2/"
    fname = f"aigefs.t{cycle_str}z.pres.{fxx_str}.grib2"
    try:
        r_idx = http_session.get(f"{url_base}{fname}.idx", timeout=8)
        if r_idx.status_code != 200:
            return None
        idx_lines = r_idx.text.splitlines()
        target_idx = None
        for i, line in enumerate(idx_lines):
            if ":HGT:500 mb:" in line or ":HGT:500:" in line:
                target_idx = i
                break
        if target_idx is None:
            return None
        start_byte = int(idx_lines[target_idx].split(':')[1])
        end_byte = int(idx_lines[target_idx + 1].split(':')[1]) - 1 if target_idx + 1 < len(idx_lines) else ""
        headers = {"Range": f"bytes={start_byte}-{end_byte}"}
        resp = http_session.get(f"{url_base}{fname}", headers=headers, timeout=12)
        if resp.status_code not in [200, 206]:
            return None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
        ds = xr.open_dataset(tmp_path, engine="cfgrib")
        grid = standardize_grid(ds)
        ds.close()
        os.remove(tmp_path)
        return grid
    except Exception:
        return None

def fetch_aigefs_ensemble(fxx):
    date_str = date_dt.strftime('%Y%m%d')
    cycle_str = date_dt.strftime('%H')
    members = [f"mem{str(i).zfill(3)}" for i in range(31)]
    def _worker(m_str):
        for sq in [":HGT:500 mb:", ":HGT:500:"]:
            try:
                H = Herbie(global_init_time, model="aigefs", product="pres", member=m_str, fxx=fxx, source="nomads")
                ds = H.xarray(sq)
                grid = standardize_grid(ds)
                if grid is not None and grid.ndim == 2:
                    return grid
            except Exception:
                continue
        return fetch_aigefs_direct_http(date_str, cycle_str, m_str, fxx)

    grids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
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


# ==============================================================================
# 6. SPREAD & UNDERLAY RENDER ENGINE (SEQUENTIAL & DIVERGING)
# ==============================================================================
SPREAD_LEVELS = np.arange(10, 130, 10)  # 10m to 120m intervals
SPREAD_CMAP = plt.cm.YlOrRd

DIFF_LEVELS = np.arange(-80, 85, 10)  # -80m to +80m difference intervals
DIFF_CMAP = plt.cm.RdBu_r
DIFF_NORM = mcolors.TwoSlopeNorm(vmin=-80, vcenter=0, vmax=80)

def compute_spread_diagnostics(gefs_mems, aigefs_mems, ifs_mems, aifs_mems):
    def calc_spread(mem_list):
        if len(mem_list) > 1:
            return np.std(np.stack(mem_list), axis=0, ddof=1)
        return None

    spr_gefs = calc_spread(gefs_mems)
    spr_aigefs = calc_spread(aigefs_mems)
    spr_ifs = calc_spread(ifs_mems)
    spr_aifs = calc_spread(aifs_mems)

    means = []
    intra_vars = []
    for mems, spr in [(gefs_mems, spr_gefs), (aigefs_mems, spr_aigefs), 
                      (ifs_mems, spr_ifs), (aifs_mems, spr_aifs)]:
        if mems:
            means.append(np.mean(np.stack(mems), axis=0))
        if spr is not None:
            intra_vars.append(spr**2)

    # Inter-Model Disagreement
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

def render_spread_underlay(spread_grid, filename, target_dir):
    if spread_grid is None or np.all(np.isnan(spread_grid)):
        return
    fig_spr = plt.figure(figsize=(12, 8), dpi=100)
    ax_spr = fig_spr.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0))
    ax_spr.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax_spr.add_feature(cfeature.LAND, facecolor="#f8f9fa")
    ax_spr.add_feature(cfeature.OCEAN, facecolor="#eef2f5")
    
    ax_spr.contourf(
        target_lons, target_lats, spread_grid,
        levels=SPREAD_LEVELS, cmap=SPREAD_CMAP,
        transform=ccrs.PlateCarree(), extend='max', alpha=0.85
    )
    
    ax_spr.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#333333")
    ax_spr.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="#333333")
    ax_spr.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#666666", linestyle=":")
    ax_spr.add_feature(cfeature.LAKES, facecolor="#eef2f5", edgecolor="#666666", linewidth=0.4)

    fig_spr.savefig(os.path.join(target_dir, filename), dpi=100)
    plt.close(fig_spr)

def render_diff_underlay(diff_grid, filename, target_dir):
    if diff_grid is None or np.all(np.isnan(diff_grid)):
        return
    fig_d = plt.figure(figsize=(12, 8), dpi=100)
    ax_d = fig_d.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0))
    ax_d.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax_d.add_feature(cfeature.LAND, facecolor="#f8f9fa")
    ax_d.add_feature(cfeature.OCEAN, facecolor="#eef2f5")

    ax_d.contourf(
        target_lons, target_lats, diff_grid,
        levels=DIFF_LEVELS, cmap=DIFF_CMAP, norm=DIFF_NORM,
        transform=ccrs.PlateCarree(), extend='both', alpha=0.85
    )

    ax_d.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#333333")
    ax_d.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="#333333")
    ax_d.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#666666", linestyle=":")
    ax_d.add_feature(cfeature.LAKES, facecolor="#eef2f5", edgecolor="#666666", linewidth=0.4)

    fig_d.savefig(os.path.join(target_dir, filename), dpi=100)
    plt.close(fig_d)


# ==============================================================================
# 7. BATCH PRODUCTION LOOP (CURRENT CYCLE)
# ==============================================================================
for fxx in FXX_STEPS:
    valid_time = date_dt + timedelta(hours=fxx)
    print(f"\n📦 Processing Lead Time F{fxx:03d} (Valid: {valid_time.strftime('%Y-%m-%d %H:%MZ')})...")
    
    # 1. Fetch ensembles
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

    # 3. Physics vs AI Difference Calculation
    phys_cluster = [m for m in [gefs_mean, ifs_mean] if m is not None]
    ai_cluster = [m for m in [aigefs_mean, aifs_mean] if m is not None]
    if phys_cluster and ai_cluster:
        physics_vs_ai_diff = np.mean(np.stack(phys_cluster), axis=0) - np.mean(np.stack(ai_cluster), axis=0)
        render_diff_underlay(physics_vs_ai_diff, f"physics_vs_ai_f{fxx:03d}.png", CURRENT_SPREAD_DIR)

    # 4. Compute and render Spread Underlays
    spread_diags = compute_spread_diagnostics(gefs_mems, aigefs_mems, ifs_mems, aifs_mems)
    for spr_key, spr_grid in spread_diags.items():
        if spr_grid is not None:
            render_spread_underlay(spr_grid, f"{spr_key}_f{fxx:03d}.png", CURRENT_SPREAD_DIR)

    # 5. Fetch deterministics
    det_gfs   = fetch_deterministic("gfs", "pgrb2.0p25", ":HGT:500 mb:", fxx)
    det_ifs   = fetch_deterministic("ifs", "oper", ":gh:500", fxx)
    det_aifs  = fetch_deterministic("aifs", "oper", ":z:500:", fxx)
    det_aigfs = fetch_aigfs(fxx)

    # 6. Extract vector contours
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

    json_path = os.path.join(CURRENT_DATA_DIR, f"f{fxx:03d}.json")
    with open(json_path, "w") as f:
        json.dump(step_data, f, separators=(',', ':'))

plt.close(fig)


# ==============================================================================
# 8. ARCHIVE PRUNING & MANIFEST GENERATION (ROLLING 10 RUNS)
# ==============================================================================
print("\n🗄️ Managing 10-Run Archive...")
all_cycles = sorted(
    [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))],
    reverse=True
)

# Prune cycles older than 10
if len(all_cycles) > 10:
    for old_cycle in all_cycles[10:]:
        old_path = os.path.join(RUNS_DIR, old_cycle)
        print(f"  🗑️ Pruning archive cycle older than 10 runs: {old_cycle}")
        shutil.rmtree(old_path, ignore_errors=True)
all_cycles = all_cycles[:10]

# Build manifest.json
manifest = {
    "latest": all_cycles[0],
    "runs": []
}
for c_id in all_cycles:
    try:
        dt = datetime.strptime(c_id, "%Y%m%d_%H")
        manifest["runs"].append({
            "id": c_id,
            "label": dt.strftime("%Y-%m-%d %Hz"),
            "init": dt.strftime("%Y-%m-%d %H:00")
        })
    except Exception:
        pass

manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"✅ Generated manifest.json with {len(manifest['runs'])} available cycles.")


# ==============================================================================
# 9. GENERATE DYNAMIC HTML FRONTEND (WITH CYCLE SCRUBBING & DPROG/DT)
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
      width: 100%; max-width: 1250px; padding: 14px 20px 6px 20px; box-sizing: border-box; text-align: center;
    }}
    .header h2 {{ margin: 0 0 4px 0; font-size: 1.35rem; }}
    .header .subtitle {{ font-size: 0.85rem; color: var(--text-muted); }}

    .toolbar {{
      width: 100%; max-width: 1250px; background: var(--card-bg); border: 1px solid var(--border);
      border-radius: 10px; padding: 10px 16px; margin: 8px 0; box-sizing: border-box;
      display: flex; flex-direction: column; gap: 10px;
    }}
    .row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 10px; justify-content: space-between; }}
    .btn {{
      background: var(--accent); color: white; border: none; padding: 5px 12px; border-radius: 6px;
      cursor: pointer; font-size: 12px; font-weight: 600; transition: 0.15s;
    }}
    .btn:hover {{ filter: brightness(1.15); }}
    .btn-secondary {{ background: #475569; }}
    .btn-micro {{ background: #334155; padding: 2px 7px; font-size: 11px; border-radius: 4px; }}
    .btn-micro:hover {{ background: #475569; }}
    
    select {{
      padding: 5px 8px; border-radius: 6px; background: #0f172a; color: white;
      border: 1px solid var(--border); font-size: 12px; cursor: pointer;
    }}

    .model-group-row {{
      display: flex; flex-direction: column; gap: 6px; border-top: 1px solid var(--border); padding-top: 8px;
    }}
    .group-line {{
      display: flex; align-items: center; gap: 12px; flex-wrap: wrap; font-size: 12px;
    }}
    .group-label {{
      font-weight: bold; min-width: 130px; display: flex; align-items: center; gap: 6px;
    }}
    .checkbox-item {{ display: flex; align-items: center; gap: 5px; cursor: pointer; margin-right: 6px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 2px; display: inline-block; }}

    .contour-section {{ border-top: 1px solid var(--border); padding-top: 6px; display: flex; flex-direction: column; gap: 6px; }}
    .contour-chips {{ display: flex; flex-wrap: wrap; gap: 4px; max-height: 70px; overflow-y: auto; }}
    .chip {{
      background: #0f172a; border: 1px solid var(--border); color: #cbd5e1;
      padding: 2px 7px; border-radius: 4px; font-size: 11px; cursor: pointer; user-select: none;
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
      color: #111; padding: 6px 10px; border-radius: 6px; font-size: 11px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.25); display: flex; flex-direction: column; gap: 3px;
      pointer-events: none; max-width: 250px;
    }}
    .legend-line {{ display: flex; align-items: center; gap: 7px; }}
    .line-sample {{ width: 20px; height: 3px; display: inline-block; }}

    .title-stamp {{
      position: absolute; top: 12px; left: 12px; background: rgba(255, 255, 255, 0.94);
      color: #111; padding: 6px 10px; border-radius: 6px; font-size: 12px; font-weight: bold;
      pointer-events: none; line-height: 1.3;
    }}
    
    .spread-colorbar {{
      display: none; align-items: center; gap: 6px; font-size: 11px;
    }}
    .colorbar-gradient {{
      width: 120px; height: 11px; border-radius: 3px; border: 1px solid var(--border);
    }}

    .footer {{ margin: 12px 0 25px 0; font-size: 12px; color: var(--text-muted); }}
    kbd {{ background: #334155; padding: 2px 5px; border-radius: 4px; font-size: 11px; }}
  </style>
</head>
<body>

  <div class="header">
    <h2>500 hPa Multi-Ensemble & Deterministic Spaghetti Viewer</h2>
    <div class="subtitle" id="cycleSubtitle">Multi-Cycle Operational Trend Analysis (10-Run Archive)</div>
  </div>

  <div class="toolbar">
    <!-- Row 1: Model Cycle Selection & dprog/dt sync -->
    <div class="row">
      <div style="display: flex; gap: 6px; align-items: center;">
        <b>Model Run:</b>
        <button class="btn btn-secondary" onclick="stepRun(-1)" title="Step to newer run (Arrow Up)">▲ Newer</button>
        <select id="runSelect" onchange="onRunChanged()"></select>
        <button class="btn btn-secondary" onclick="stepRun(1)" title="Step to older run (Arrow Down)">▼ Older</button>
      </div>

      <div style="display: flex; gap: 10px; align-items: center; font-size: 12px;">
        <label title="When changing model runs, automatically shifts lead time to track the exact same valid time.">
          <input type="checkbox" id="chk_dprogdt" checked> <b>Sync Valid Time (dprog/dt)</b>
        </label>
      </div>

      <div style="display: flex; gap: 8px; align-items: center;">
        <b>Underlay:</b>
        <select id="bgSelect" onchange="updateBackground()">
          <option value="none">Basemap Only</option>
          <option value="physics_vs_ai">Physics minus AI (Phys - AI)</option>
          <option value="inter">Inter-Model Disagreement (σ_inter)</option>
          <option value="total">Total Super Spread (σ_total)</option>
          <option value="intra_pooled">Pooled Internal Chaos (σ_intra)</option>
          <option value="gefs">GEFS Spread</option>
          <option value="aigefs">AIGEFS Spread</option>
          <option value="ifs">ECMWF IFS Spread</option>
          <option value="aifs">AIFS Spread</option>
        </select>

        <div class="spread-colorbar" id="cbarBox">
          <span id="cbarMin">10m</span>
          <div class="colorbar-gradient" id="cbarGrad"></div>
          <span id="cbarMax">120m+</span>
        </div>
      </div>
    </div>

    <!-- Row 2: Lead Scrubbing & Display Mode -->
    <div class="row">
      <div style="display: flex; gap: 6px; align-items: center;">
        <b>Forecast Lead:</b>
        <button class="btn btn-secondary" onclick="stepLead(-1)">◀ Prev</button>
        <select id="leadSelect" onchange="loadStepData()">
          {"".join([f'<option value="{f:03d}">F{f:03d} (+{f}h)</option>' for f in FXX_STEPS])}
        </select>
        <button class="btn btn-secondary" onclick="stepLead(1)">Next ▶</button>
      </div>

      <div style="display: flex; gap: 12px; align-items: center; font-size: 12px;">
        <b>Display Mode:</b>
        <label><input type="radio" name="displayMode" value="spaghetti" checked onchange="draw()"> Spaghetti + Means</label>
        <label><input type="radio" name="displayMode" value="means" onchange="draw()"> Means Only</label>
      </div>
    </div>

    <!-- Row 3: Structured Model Toggles (Ensembles vs Deterministics) -->
    <div class="model-group-row">
      <div class="group-line">
        <span class="group-label">
          Ensembles & Means:
          <button class="btn btn-micro" onclick="setGroup('ens', true)">All</button>
          <button class="btn btn-micro" onclick="setGroup('ens', false)">None</button>
        </span>
        <label class="checkbox-item"><input type="checkbox" id="chk_super" checked onchange="draw()"><span class="swatch" style="background:#000;"></span> Super Mean</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_gefs" checked onchange="draw()"><span class="swatch" style="background:#0044ff;"></span> GEFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aigefs" checked onchange="draw()"><span class="swatch" style="background:#8e44ad;"></span> AIGEFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_ifs" checked onchange="draw()"><span class="swatch" style="background:#d63031;"></span> ECMWF IFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aifs" checked onchange="draw()"><span class="swatch" style="background:#00b894;"></span> AIFS</label>
      </div>

      <div class="group-line">
        <span class="group-label">
          Deterministics:
          <button class="btn btn-micro" onclick="setGroup('det', true)">All</button>
          <button class="btn btn-micro" onclick="setGroup('det', false)">None</button>
        </span>
        <label class="checkbox-item"><input type="checkbox" id="chk_gfs" checked onchange="draw()"><span class="swatch" style="background:#001155; border-bottom: 2px dashed #fff;"></span> GFS Det</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aigfs" checked onchange="draw()"><span class="swatch" style="background:#4a235a; border-bottom: 2px dashed #fff;"></span> AIGFS Det</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_ecmwf_det" onchange="draw()"><span class="swatch" style="background:#78281f; border-bottom: 2px dashed #fff;"></span> IFS Det</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aifs_det" onchange="draw()"><span class="swatch" style="background:#145a32; border-bottom: 2px dashed #fff;"></span> AIFS Det</label>
      </div>
    </div>

    <!-- Row 4: Height Contours (4980 to 6000m) -->
    <div class="contour-section">
      <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
        <span style="font-size: 12px; font-weight: bold;">Contour Levels (m):</span>
        <button class="btn btn-secondary" style="padding: 2px 7px; font-size: 11px;" onclick="selectAllContours(true)">Select All</button>
        <button class="btn btn-secondary" style="padding: 2px 7px; font-size: 11px;" onclick="setStandardContours()">Standard 500mb (5400-5880)</button>
        <button class="btn btn-secondary" style="padding: 2px 7px; font-size: 11px;" onclick="selectAllContours(false)">Clear All</button>
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
    Shortcuts: Use <kbd>←</kbd> and <kbd>→</kbd> for Forecast Leads | Use <kbd>↑</kbd> (Newer) and <kbd>↓</kbd> (Older) for Model Cycles.
  </div>

  <script>
    const ALL_CONTOURS = {ALL_CONTOURS};
    const STANDARD_CONTOURS = [5400, 5520, 5640, 5760, 5880];
    const FXX_STEPS = {FXX_STEPS};
    let selectedContours = new Set(ALL_CONTOURS);
    let dataCache = {{}};
    let currentStepData = null;
    let manifestData = {{ runs: [] }};

    function parseUtcDate(str) {{
      const [d, t] = str.split(' ');
      const [yr, mo, da] = d.split('-').map(Number);
      const [hr, mi] = t.split(':').map(Number);
      return new Date(Date.UTC(yr, mo - 1, da, hr, mi));
    }}

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

    function setGroup(grp, state) {{
      if (grp === 'ens') {{
        ['chk_super', 'chk_gefs', 'chk_aigefs', 'chk_ifs', 'chk_aifs'].forEach(id => {{
          document.getElementById(id).checked = state;
        }});
      }} else if (grp === 'det') {{
        ['chk_gfs', 'chk_aigfs', 'chk_ecmwf_det', 'chk_aifs_det'].forEach(id => {{
          document.getElementById(id).checked = state;
        }});
      }}
      draw();
    }}

    // Underlay & Colorbar Styling
    function updateBackground() {{
      const bg = document.getElementById('bgSelect').value;
      const fxx = document.getElementById('leadSelect').value;
      const runId = document.getElementById('runSelect').value;
      const img = document.getElementById('mapUnderlay');
      const cbar = document.getElementById('cbarBox');
      const cbarGrad = document.getElementById('cbarGrad');
      const cbarMin = document.getElementById('cbarMin');
      const cbarMax = document.getElementById('cbarMax');

      if (bg === 'none') {{
        img.src = 'basemap.png';
        cbar.style.display = 'none';
      }} else if (bg === 'physics_vs_ai') {{
        img.src = `runs/${{runId}}/spread/physics_vs_ai_f${{fxx}}.png`;
        cbar.style.display = 'flex';
        cbarGrad.style.background = 'linear-gradient(to right, #2166ac, #67a9cf, #d1e5f0, #f7f7f7, #fddbc7, #ef8a62, #b2182b)';
        cbarMin.textContent = '-80m (AI)';
        cbarMax.textContent = '+80m (Phys)';
      }} else {{
        img.src = `runs/${{runId}}/spread/${{bg}}_f${{fxx}}.png`;
        cbar.style.display = 'flex';
        cbarGrad.style.background = 'linear-gradient(to right, #ffffcc, #fed976, #feb24c, #fd8d3c, #f03b20, #bd0026)';
        cbarMin.textContent = '10m';
        cbarMax.textContent = '120m+';
      }}
    }}

    // Load Data
    async function loadStepData() {{
      const runId = document.getElementById('runSelect').value;
      const fxx = document.getElementById('leadSelect').value;
      const cacheKey = `${{runId}}_${{fxx}}`;
      updateBackground();

      if (dataCache[cacheKey]) {{
        currentStepData = dataCache[cacheKey];
        draw();
        return;
      }}
      try {{
        const resp = await fetch(`runs/${{runId}}/data/f${{fxx}}.json`);
        const json = await resp.json();
        dataCache[cacheKey] = json;
        currentStepData = json;
        draw();
      }} catch (err) {{
        console.error("Error loading JSON for lead F" + fxx, err);
      }}
    }}

    // Navigation & dprog/dt Sync
    function stepLead(delta) {{
      const select = document.getElementById('leadSelect');
      let idx = select.selectedIndex + delta;
      if (idx >= 0 && idx < select.options.length) {{
        select.selectedIndex = idx;
        loadStepData();
      }}
    }}

    function stepRun(delta) {{
      const select = document.getElementById('runSelect');
      let idx = select.selectedIndex + delta;
      if (idx >= 0 && idx < select.options.length) {{
        select.selectedIndex = idx;
        onRunChanged();
      }}
    }}

    function onRunChanged() {{
      const syncDprog = document.getElementById('chk_dprogdt').checked;
      if (syncDprog && currentStepData) {{
        const currentValid = parseUtcDate(currentStepData.valid);
        const selectedRunObj = manifestData.runs.find(r => r.id === document.getElementById('runSelect').value);
        if (selectedRunObj) {{
          const newInit = parseUtcDate(selectedRunObj.init);
          const targetFxx = Math.round((currentValid - newInit) / 3600000);
          
          if (FXX_STEPS.includes(targetFxx)) {{
            document.getElementById('leadSelect').value = String(targetFxx).padStart(3, '0');
          }}
        }}
      }}
      loadStepData();
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowLeft') {{ e.preventDefault(); stepLead(-1); }}
      if (e.key === 'ArrowRight') {{ e.preventDefault(); stepLead(1); }}
      if (e.key === 'ArrowUp') {{ e.preventDefault(); stepRun(-1); }}    // Newer cycle
      if (e.key === 'ArrowDown') {{ e.preventDefault(); stepRun(1); }} // Older cycle
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
          currentStepData.gefs_members.forEach(m => drawLines(m, '#2980b9', 0.9, 0.55));
          legendItems.push({{ label: `GEFS Members (${{currentStepData.gefs_members.length}})`, color: '#2980b9', dash: false }});
        }}
        if (showAigefs && currentStepData.aigefs_members) {{
          currentStepData.aigefs_members.forEach(m => drawLines(m, '#9b59b6', 0.9, 0.55));
          legendItems.push({{ label: `AIGEFS Members (${{currentStepData.aigefs_members.length}})`, color: '#9b59b6', dash: false }});
        }}
        if (showIfs && currentStepData.ifs_members) {{
          currentStepData.ifs_members.forEach(m => drawLines(m, '#e74c3c', 0.9, 0.55));
          legendItems.push({{ label: `IFS Members (${{currentStepData.ifs_members.length}})`, color: '#e74c3c', dash: false }});
        }}
        if (showAifs && currentStepData.aifs_members) {{
          currentStepData.aifs_members.forEach(m => drawLines(m, '#00b894', 0.9, 0.55));
          legendItems.push({{ label: `AIFS Members (${{currentStepData.aifs_members.length}})`, color: '#00b894', dash: false }});
        }}
      }}

      // 2. Deterministics (Dashed)
      if (showGfs && currentStepData.gfs_det) {{
        drawLines(currentStepData.gfs_det, '#001155', 2.0, 1.0, true);
        legendItems.push({{ label: 'GFS Det', color: '#001155', dash: true }});
      }}
      if (showAigfs && currentStepData.aigfs_det) {{
        drawLines(currentStepData.aigfs_det, '#4a235a', 2.0, 1.0, true);
        legendItems.push({{ label: 'AIGFS Det', color: '#4a235a', dash: true }});
      }}
      if (showEcmwf && currentStepData.ecmwf_det) {{
        drawLines(currentStepData.ecmwf_det, '#78281f', 2.0, 1.0, true);
        legendItems.push({{ label: 'IFS Det', color: '#78281f', dash: true }});
      }}
      if (showAifsDet && currentStepData.aifs_det) {{
        drawLines(currentStepData.aifs_det, '#145a32', 2.0, 1.0, true);
        legendItems.push({{ label: 'AIFS Det', color: '#145a32', dash: true }});
      }}

      // 3. Ensemble Means
      if (showGefs && currentStepData.gefs_mean) {{
        drawLines(currentStepData.gefs_mean, '#0044ff', 2.4, 1.0);
        legendItems.push({{ label: 'GEFS Mean', color: '#0044ff', dash: false }});
      }}
      if (showAigefs && currentStepData.aigefs_mean) {{
        drawLines(currentStepData.aigefs_mean, '#8e44ad', 2.4, 1.0);
        legendItems.push({{ label: 'AIGEFS Mean', color: '#8e44ad', dash: false }});
      }}
      if (showIfs && currentStepData.ifs_mean) {{
        drawLines(currentStepData.ifs_mean, '#d63031', 2.4, 1.0);
        legendItems.push({{ label: 'IFS Mean', color: '#d63031', dash: false }});
      }}
      if (showAifs && currentStepData.aifs_mean) {{
        drawLines(currentStepData.aifs_mean, '#00b894', 2.4, 1.0);
        legendItems.push({{ label: 'AIFS Mean', color: '#00b894', dash: false }});
      }}

      // 4. Superensemble Mean
      if (showSuper && currentStepData.super_mean) {{
        drawLines(currentStepData.super_mean, '#000000', 3.2, 1.0);
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

    // Initial Load & Populate Runs Dropdown from manifest.json
    async function initViewer() {{
      try {{
        const resp = await fetch('manifest.json');
        manifestData = await resp.json();
        const runSelect = document.getElementById('runSelect');
        runSelect.innerHTML = manifestData.runs.map((r, i) => 
          `<option value="${{r.id}}">${{r.label}}${{i === 0 ? ' (Latest)' : ''}}</option>`
        ).join('');
        runSelect.value = manifestData.latest;
      }} catch (e) {{
        console.warn("Could not load manifest.json, using fallback.");
      }}
      loadStepData();
    }}

    initViewer();
  </script>
</body>
</html>
"""

with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
    f.write(html_content)

print(f"🎉 Production build complete! Files written to {OUTPUT_DIR}")
