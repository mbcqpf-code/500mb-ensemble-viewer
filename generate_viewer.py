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
# 2. DYNAMIC CYCLE DETECTOR & IDEMPOTENCY CHECK
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

CURRENT_RUN_DIR = os.path.join(RUNS_DIR, current_cycle_id)
CURRENT_DATA_DIR = os.path.join(CURRENT_RUN_DIR, "data")
CURRENT_SPREAD_DIR = os.path.join(CURRENT_RUN_DIR, "spread")
CURRENT_GRIDS_DIR = os.path.join(CURRENT_RUN_DIR, "grids")
os.makedirs(CURRENT_DATA_DIR, exist_ok=True)
os.makedirs(CURRENT_SPREAD_DIR, exist_ok=True)
os.makedirs(CURRENT_GRIDS_DIR, exist_ok=True)

# Idempotency Check
skip_download_and_generation = False
if os.path.exists(CURRENT_DATA_DIR) and os.path.exists(CURRENT_SPREAD_DIR):
    all_json_exist = all(os.path.exists(os.path.join(CURRENT_DATA_DIR, f"f{f:03d}.json")) for f in FXX_STEPS)
    all_png_exist = all(os.path.exists(os.path.join(CURRENT_SPREAD_DIR, f"total_f{f:03d}.png")) for f in FXX_STEPS)
    metrics_exist = os.path.exists(os.path.join(CURRENT_RUN_DIR, "consensus_metrics.json"))
    if all_json_exist and all_png_exist and metrics_exist:
        skip_download_and_generation = True
        print(f"⚡ Cycle {current_cycle_id} already exists with complete 240h data!")
        print("⏩ Skipping download and grid extraction. Proceeding directly to archive and frontend updates.\n")

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
# 6. SPREAD & TREND UNDERLAY RENDER ENGINES
# ==============================================================================
SPREAD_LEVELS = np.arange(10, 130, 10)
SPREAD_CMAP = plt.cm.YlOrRd

DIFF_LEVELS = np.arange(-80, 85, 10)
DIFF_CMAP = plt.cm.RdBu_r
DIFF_NORM = mcolors.TwoSlopeNorm(vmin=-80, vcenter=0, vmax=80)

CONV_LEVELS = np.arange(-30, 35, 5)
CONV_CMAP = plt.cm.PRGn_r
CONV_NORM = mcolors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)

FLIP_LEVELS = np.arange(10, 90, 10)
FLIP_CMAP = plt.cm.YlOrRd

DET_OUTLIER_PALETTES = {
    'gfs': ['#90caf9', '#42a5f5', '#1e88e5', '#0d47a1'],
    'aigfs': ['#e1bee7', '#ba68c8', '#8e24aa', '#4a148c'],
    'ecmwf': ['#ffcdd2', '#e57373', '#e53935', '#b71c1c'],
    'aifs': ['#c8e6c9', '#81c784', '#43a047', '#1b5e20']
}
DET_OUTLIER_COLORS = {
    'gfs': '#001155',
    'aigfs': '#4a235a',
    'ecmwf': '#78281f',
    'aifs': '#145a32'
}

ENS_OUTLIER_PALETTES = {
    'gefs': ['#90caf9', '#42a5f5', '#1e88e5', '#0d47a1'],
    'aigefs': ['#e1bee7', '#ba68c8', '#8e24aa', '#4a148c'],
    'ifs': ['#ffcdd2', '#e57373', '#e53935', '#b71c1c'],
    'aifs': ['#c8e6c9', '#81c784', '#43a047', '#1b5e20']
}
ENS_OUTLIER_COLORS = {
    'gefs': '#0044ff',
    'aigefs': '#8e44ad',
    'ifs': '#d63031',
    'aifs': '#00b894'
}

def compute_consensus_metrics(model_grid, super_mean):
    """Computes centered spatial pattern correlation (r) and spatial RMS error (meters)."""
    if model_grid is None or super_mean is None or np.all(np.isnan(model_grid)) or np.all(np.isnan(super_mean)):
        return {"r": None, "rmsd": None}
    try:
        m_prime = model_grid - np.nanmean(model_grid)
        s_prime = super_mean - np.nanmean(super_mean)
        num = np.nansum(m_prime * s_prime)
        denom = np.sqrt(np.nansum(m_prime**2) * np.nansum(s_prime**2)) + 1e-6
        r_val = float(np.clip(num / denom, -1.0, 1.0))
        rmsd_val = float(np.sqrt(np.nanmean((model_grid - super_mean)**2)))
        return {"r": round(r_val, 3), "rmsd": round(rmsd_val, 1)}
    except Exception:
        return {"r": None, "rmsd": None}

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

    if len(means) > 1:
        spr_inter = np.std(np.stack(means), axis=0, ddof=1)
    else:
        spr_inter = np.zeros_like(target_lats)

    if intra_vars:
        spr_intra_pooled = np.sqrt(np.mean(np.stack(intra_vars), axis=0))
    else:
        spr_intra_pooled = np.zeros_like(target_lats)

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

def render_contourf_underlay(grid, levels, cmap, norm, filename, target_dir, extend='both'):
    fig_u = plt.figure(figsize=(12, 8), dpi=100)
    ax_u = fig_u.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0))
    ax_u.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax_u.add_feature(cfeature.LAND, facecolor="#f8f9fa")
    ax_u.add_feature(cfeature.OCEAN, facecolor="#eef2f5")

    if grid is not None and not np.all(np.isnan(grid)) and not np.all(grid == 0):
        if norm is not None:
            ax_u.contourf(target_lons, target_lats, grid, levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend=extend, alpha=0.85)
        else:
            ax_u.contourf(target_lons, target_lats, grid, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), extend=extend, alpha=0.85)

    ax_u.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#333333")
    ax_u.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="#333333")
    ax_u.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#666666", linestyle=":")
    ax_u.add_feature(cfeature.LAKES, facecolor="#eef2f5", edgecolor="#666666", linewidth=0.4)

    fig_u.savefig(os.path.join(target_dir, filename), dpi=100)
    plt.close(fig_u)

def render_deterministic_outliers(det_dict, super_mean, spr_total, filename, target_dir):
    fig_o = plt.figure(figsize=(12, 8), dpi=100)
    ax_o = fig_o.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0))
    ax_o.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax_o.add_feature(cfeature.LAND, facecolor="#f8f9fa")
    ax_o.add_feature(cfeature.OCEAN, facecolor="#eef2f5")

    valid_models = {k: v for k, v in det_dict.items() if v is not None}
    if super_mean is not None and spr_total is not None and valid_models:
        eps = 2.0
        z_dict = {k: (v - super_mean) / (spr_total + eps) for k, v in valid_models.items()}
        abs_z_dict = {k: np.abs(z) for k, z in z_dict.items()}
        
        model_keys = list(abs_z_dict.keys())
        abs_stack = np.stack([abs_z_dict[k] for k in model_keys], axis=0)
        dom_idx = np.argmax(abs_stack, axis=0)
        max_abs_z = np.max(abs_stack, axis=0)
        
        for i, k in enumerate(model_keys):
            mask_dom = (dom_idx == i) & (max_abs_z >= 2.0)
            if np.any(mask_dom):
                z_dom = np.where(mask_dom, max_abs_z, np.nan)
                palette = DET_OUTLIER_PALETTES.get(k, ['#fed976', '#feb24c', '#fd8d3c', '#f03b20'])
                try:
                    ax_o.contourf(
                        target_lons, target_lats, z_dom,
                        levels=[2.0, 2.5, 3.0, 4.0],
                        colors=palette,
                        transform=ccrs.PlateCarree(),
                        extend='max', alpha=0.85
                    )
                except Exception:
                    pass

        for k, abs_z in abs_z_dict.items():
            if np.any(abs_z >= 2.0):
                line_color = DET_OUTLIER_COLORS.get(k, '#000000')
                try:
                    ax_o.contour(
                        target_lons, target_lats, abs_z,
                        levels=[2.0],
                        colors=[line_color],
                        linewidths=1.8,
                        linestyles='--',
                        transform=ccrs.PlateCarree()
                    )
                except Exception:
                    pass

    ax_o.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#333333")
    ax_o.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="#333333")
    ax_o.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#666666", linestyle=":")
    ax_o.add_feature(cfeature.LAKES, facecolor="#eef2f5", edgecolor="#666666", linewidth=0.4)

    fig_o.savefig(os.path.join(target_dir, filename), dpi=100)
    plt.close(fig_o)

def render_ensemble_lone_wolf(means_dict, spr_inter, filename, target_dir):
    fig_e = plt.figure(figsize=(12, 8), dpi=100)
    ax_e = fig_e.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0))
    ax_e.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax_e.add_feature(cfeature.LAND, facecolor="#f8f9fa")
    ax_e.add_feature(cfeature.OCEAN, facecolor="#eef2f5")

    keys = ['gefs', 'aigefs', 'ifs', 'aifs']
    if all(means_dict.get(k) is not None for k in keys) and spr_inter is not None:
        stack = np.stack([means_dict[k] for k in keys], axis=0)
        super_m = np.mean(stack, axis=0)
        
        sq_devs = (stack - super_m)**2
        total_sq = np.sum(sq_devs, axis=0) + 1e-5
        var_shares = sq_devs / total_sq
        mask_active = spr_inter >= 15.0
        
        for idx, k in enumerate(keys):
            other_indices = [j for j in range(4) if j != idx]
            trio_mean = np.mean(stack[other_indices], axis=0)
            delta_3v1 = stack[idx] - trio_mean
            abs_delta = np.abs(delta_3v1)
            is_rogue = mask_active & (var_shares[idx] >= 0.60) & (abs_delta >= 25.0)
            
            if np.any(is_rogue):
                grid_rogue = np.where(is_rogue, abs_delta, np.nan)
                palette = ENS_OUTLIER_PALETTES.get(k, ['#fed976', '#feb24c', '#fd8d3c', '#f03b20'])
                try:
                    ax_e.contourf(
                        target_lons, target_lats, grid_rogue,
                        levels=[25.0, 35.0, 50.0, 75.0],
                        colors=palette,
                        transform=ccrs.PlateCarree(),
                        extend='max', alpha=0.85
                    )
                    ax_e.contour(
                        target_lons, target_lats, abs_delta,
                        levels=[25.0],
                        colors=[ENS_OUTLIER_COLORS.get(k, '#000000')],
                        linewidths=1.8,
                        transform=ccrs.PlateCarree()
                    )
                except Exception:
                    pass

    ax_e.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#333333")
    ax_e.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="#333333")
    ax_e.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#666666", linestyle=":")
    ax_e.add_feature(cfeature.LAKES, facecolor="#eef2f5", edgecolor="#666666", linewidth=0.4)

    fig_e.savefig(os.path.join(target_dir, filename), dpi=100)
    plt.close(fig_e)


# ==============================================================================
# 7. BATCH PRODUCTION LOOP (ONLY EXECUTED IF RUN IS NOT ALREADY CACHED)
# ==============================================================================
all_run_metrics = {}

if not skip_download_and_generation:
    for fxx in FXX_STEPS:
        valid_time = date_dt + timedelta(hours=fxx)
        fxx_key = f"{fxx:03d}"
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

        # 3. Fetch deterministics
        det_gfs   = fetch_deterministic("gfs", "pgrb2.0p25", ":HGT:500 mb:", fxx)
        det_ifs   = fetch_deterministic("ifs", "oper", ":gh:500", fxx)
        det_aifs  = fetch_deterministic("aifs", "oper", ":z:500:", fxx)
        det_aigfs = fetch_aigfs(fxx)

        # 4. Consensus Metrics (Pattern correlation r and RMSD vs Super Mean)
        step_metrics = {
            "gefs_mean": compute_consensus_metrics(gefs_mean, super_mean),
            "aigefs_mean": compute_consensus_metrics(aigefs_mean, super_mean),
            "ifs_mean": compute_consensus_metrics(ifs_mean, super_mean),
            "aifs_mean": compute_consensus_metrics(aifs_mean, super_mean),
            "gfs_det": compute_consensus_metrics(det_gfs, super_mean),
            "aigfs_det": compute_consensus_metrics(det_aigfs, super_mean),
            "ecmwf_det": compute_consensus_metrics(det_ifs, super_mean),
            "aifs_det": compute_consensus_metrics(det_aifs, super_mean)
        }
        all_run_metrics[fxx_key] = step_metrics

        # 5. Spread Diagnostics
        spread_diags = compute_spread_diagnostics(gefs_mems, aigefs_mems, ifs_mems, aifs_mems)

        # 6. Save compact 2D grids for trend / archive calculations
        if super_mean is not None and spread_diags['inter'] is not None:
            np.savez_compressed(
                os.path.join(CURRENT_GRIDS_DIR, f"f{fxx:03d}.npz"),
                super_mean=super_mean.astype(np.float32),
                spr_inter=spread_diags['inter'].astype(np.float32)
            )

        # 7. Outlier Underlays
        det_dict = {'gfs': det_gfs, 'aigfs': det_aigfs, 'ecmwf': det_ifs, 'aifs': det_aifs}
        render_deterministic_outliers(det_dict, super_mean, spread_diags.get('total'), f"outlier_det_f{fxx:03d}.png", CURRENT_SPREAD_DIR)

        means_dict = {'gefs': gefs_mean, 'aigefs': aigefs_mean, 'ifs': ifs_mean, 'aifs': aifs_mean}
        render_ensemble_lone_wolf(means_dict, spread_diags.get('inter'), f"outlier_ens_f{fxx:03d}.png", CURRENT_SPREAD_DIR)

        # 8. Physics vs AI Difference Underlay
        phys_cluster = [m for m in [gefs_mean, ifs_mean] if m is not None]
        ai_cluster = [m for m in [aigefs_mean, aifs_mean] if m is not None]
        physics_vs_ai_diff = np.mean(np.stack(phys_cluster), axis=0) - np.mean(np.stack(ai_cluster), axis=0) if phys_cluster and ai_cluster else None
        render_contourf_underlay(physics_vs_ai_diff, DIFF_LEVELS, DIFF_CMAP, DIFF_NORM, f"physics_vs_ai_f{fxx:03d}.png", CURRENT_SPREAD_DIR, extend='both')

        # 9. Standard Spread Underlays
        for spr_key, spr_grid in spread_diags.items():
            render_contourf_underlay(spr_grid, SPREAD_LEVELS, SPREAD_CMAP, None, f"{spr_key}_f{fxx:03d}.png", CURRENT_SPREAD_DIR, extend='max')

        # 10. Multi-Run Trend Diagnostics
        prior_cycles = sorted(
            [d for d in os.listdir(RUNS_DIR) if d < current_cycle_id and os.path.isdir(os.path.join(RUNS_DIR, d))],
            reverse=True
        )

        past_means = [super_mean] if super_mean is not None else []
        past_spr_inters = [spread_diags['inter']] if spread_diags['inter'] is not None else []

        for past_id in prior_cycles[:3]:
            try:
                past_dt = datetime.strptime(past_id, "%Y%m%d_%H")
                past_fxx = int(round((valid_time - past_dt).total_seconds() / 3600))
                if 0 <= past_fxx <= 240 and past_fxx % 12 == 0:
                    past_npz_path = os.path.join(RUNS_DIR, past_id, "grids", f"f{past_fxx:03d}.npz")
                    if os.path.exists(past_npz_path):
                        with np.load(past_npz_path) as data:
                            past_means.append(data['super_mean'])
                            past_spr_inters.append(data['spr_inter'])
            except Exception:
                pass

        delta_spr_inter = past_spr_inters[0] - past_spr_inters[1] if len(past_spr_inters) >= 2 else None
        render_contourf_underlay(delta_spr_inter, CONV_LEVELS, CONV_CMAP, CONV_NORM, f"convergence_f{fxx:03d}.png", CURRENT_SPREAD_DIR, extend='both')

        if len(past_means) >= 2:
            increments = [past_means[i] - past_means[i+1] for i in range(len(past_means) - 1)]
            delta_net = past_means[0] - past_means[-1]
            path_length = np.sum([np.abs(inc) for inc in increments], axis=0)
            mono_ratio = delta_net / (path_length + 1e-5)
            monotonic_trend = delta_net * np.abs(mono_ratio)
            flip_flop_energy = path_length - np.abs(delta_net)
        else:
            monotonic_trend = None
            flip_flop_energy = None

        render_contourf_underlay(monotonic_trend, DIFF_LEVELS, DIFF_CMAP, DIFF_NORM, f"monotonic_trend_f{fxx:03d}.png", CURRENT_SPREAD_DIR, extend='both')
        render_contourf_underlay(flip_flop_energy, FLIP_LEVELS, FLIP_CMAP, None, f"flipflop_f{fxx:03d}.png", CURRENT_SPREAD_DIR, extend='max')

        # 11. Extract vector contours
        super_contours = extract_contours(super_mean)
        print(f"  ↳ Extracted {len(super_contours)} height levels for Superensemble Mean.")

        step_data = {
            "fxx": fxx,
            "init": global_init_time,
            "valid": valid_time.strftime('%Y-%m-%d %H:%MZ'),
            "metrics": step_metrics,
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

    # Save aggregated consensus metrics for full lead-time tracking
    metrics_summary_path = os.path.join(CURRENT_RUN_DIR, "consensus_metrics.json")
    with open(metrics_summary_path, "w") as f:
        json.dump({"init": global_init_time, "metrics": all_run_metrics}, f, indent=2)
    print(f"✅ Saved consensus_metrics.json for cycle {current_cycle_id}")

plt.close(fig)


# ==============================================================================
# 8. ARCHIVE PRUNING & MANIFEST GENERATION (ROLLING 10 RUNS)
# ==============================================================================
print("\n🗄️ Managing 10-Run Archive...")
all_cycles = sorted(
    [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))],
    reverse=True
)

if len(all_cycles) > 10:
    for old_cycle in all_cycles[10:]:
        old_path = os.path.join(RUNS_DIR, old_cycle)
        print(f"  🗑️ Pruning archive cycle older than 10 runs: {old_cycle}")
        shutil.rmtree(old_path, ignore_errors=True)
all_cycles = all_cycles[:10]

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
# 9. GENERATE DYNAMIC HTML FRONTEND (WITH MODAL CHART & LIVE CONSENSUS METRICS)
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
    optgroup {{
      background: #1e293b; color: #94a3b8; font-weight: bold; font-style: normal;
    }}
    option {{
      background: #0f172a; color: #f8fafc; font-weight: normal;
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
      background: #f8f9fa; border-radius: 10px; overflow: hidden; box-shadow: 0 12px 30px rgba(0,0,0,0.5);
    }}
    .viewer-card img, .viewer-card canvas {{
      position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    }}
    #contourCanvas {{ pointer-events: none; }}
    
    .legend-box {{
      position: absolute; top: 12px; right: 12px; background: rgba(255, 255, 255, 0.95);
      color: #111; padding: 6px 10px; border-radius: 6px; font-size: 11px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.25); display: flex; flex-direction: column; gap: 3px;
      pointer-events: none; max-width: 320px;
    }}
    .legend-line {{ display: flex; align-items: center; gap: 7px; }}
    .line-sample {{ width: 20px; height: 3px; display: inline-block; flex-shrink: 0; }}

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

    .tooltip-container {{
      position: relative; display: inline-flex; align-items: center;
    }}
    .info-btn {{
      background: #334155; color: #93c5fd; border: none; width: 22px; height: 22px;
      border-radius: 50%; font-size: 12px; font-weight: bold; cursor: pointer;
      display: flex; align-items: center; justify-content: center; transition: 0.15s;
    }}
    .info-btn:hover {{
      background: #475569; color: #ffffff;
    }}
    .tooltip-card {{
      display: none; position: absolute; top: 30px; right: 0; width: 320px;
      background: #0f172a; border: 1px solid #475569; border-radius: 8px;
      padding: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.6); z-index: 100;
      color: #e2e8f0; font-size: 12px; line-height: 1.4;
    }}
    .tooltip-card.show {{
      display: block;
    }}
    .tooltip-title {{
      font-weight: bold; font-size: 13px; color: #60a5fa; margin-bottom: 6px;
      border-bottom: 1px solid #334155; padding-bottom: 4px;
    }}
    .tooltip-section {{ margin-top: 6px; }}
    .tooltip-section b {{ color: #cbd5e1; }}
    .underlay-desc-bar {{
      font-size: 11.5px; color: #94a3b8; font-style: italic; margin-top: -4px;
    }}

    /* Modal Overlay Styles for Consensus Tracker Chart */
    .modal-overlay {{
      display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0, 0, 0, 0.75); z-index: 9999; justify-content: center; align-items: center;
    }}
    .modal-overlay.open {{ display: flex; }}
    .modal-content {{
      background: #1e293b; border: 1px solid var(--border); border-radius: 12px;
      width: 90%; max-width: 920px; padding: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.7);
      display: flex; flex-direction: column; gap: 14px; color: var(--text);
    }}
    .modal-header {{
      display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
    }}
    .modal-header h3 {{ margin: 0; font-size: 1.15rem; color: #60a5fa; }}
    .modal-close {{
      background: transparent; border: none; font-size: 24px; color: #94a3b8; cursor: pointer; line-height: 1;
    }}
    .modal-close:hover {{ color: #ffffff; }}
    .modal-controls {{
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
      background: #0f172a; padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border);
    }}
    .chart-wrapper {{
      width: 100%; background: #0b1120; border-radius: 8px; border: 1px solid var(--border);
      padding: 10px; box-sizing: border-box; display: flex; justify-content: center;
    }}
    .modal-legend {{
      display: flex; flex-wrap: wrap; gap: 14px; font-size: 11.5px; justify-content: center; padding-top: 4px;
    }}

    .footer {{ margin: 12px 0 25px 0; font-size: 12px; color: var(--text-muted); }}
    kbd {{ background: #334155; padding: 2px 5px; border-radius: 4px; font-size: 11px; }}
  </style>
</head>
<body>

  <div class="header">
    <h2>500 hPa Multi-Ensemble & Deterministic Spaghetti Viewer</h2>
    <div class="subtitle">Multi-Cycle Operational Trend Analysis (10-Run Archive)</div>
  </div>

  <div class="toolbar">
    <!-- Row 1: Model Cycle Selection & dprog/dt Sync -->
    <div class="row">
      <div style="display: flex; gap: 6px; align-items: center;">
        <b>Model Run:</b>
        <button class="btn btn-secondary" onclick="stepRun(-1)" title="Step to newer run (Arrow Up)">▲ Newer</button>
        <select id="runSelect" onchange="onRunChanged()"></select>
        <button class="btn btn-secondary" onclick="stepRun(1)" title="Step to older run (Arrow Down)">▼ Older</button>
      </div>

      <div style="display: flex; gap: 8px; align-items: center; font-size: 12px; background: #0f172a; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border);">
        <label style="cursor: pointer; display: flex; align-items: center; gap: 6px;" title="Automatically shifts lead time when changing cycles to track the exact same valid time.">
          <input type="checkbox" id="chk_dprogdt" checked onchange="updateDprogBadge()"> 
          <span style="color: #60a5fa; font-weight: bold;">dprog/dt Mode:</span> Sync Valid Time
        </label>
        <span id="dprogStatus" style="color: #cbd5e1; font-size: 11px; margin-left: 4px;"></span>
      </div>

      <!-- Reorganized Underlay Selection with Tooltip -->
      <div style="display: flex; gap: 8px; align-items: center;">
        <b>Underlay:</b>
        <select id="bgSelect" onchange="updateBackground()">
          <optgroup label="Standard Basemap">
            <option value="none">Basemap Only (Clean)</option>
          </optgroup>
          <optgroup label="Consensus & Outliers">
            <option value="outlier_ens">Ensemble Mean Outliers (3-vs-1)</option>
            <option value="outlier_det">Deterministic Outliers (|Z| ≥ 2.0σ)</option>
            <option value="inter">Inter-Model Disagreement (σ_inter)</option>
            <option value="physics_vs_ai">Physics minus AI (Phys - AI)</option>
          </optgroup>
          <optgroup label="Multi-Run Trends (dprog/dt)">
            <option value="monotonic_trend">4-Run Monotonic Trend (Deep vs Build)</option>
            <option value="flipflop">4-Run Flip-Flop Volatility (Oscillation)</option>
            <option value="convergence">Model Convergence Rate (Δσ_inter)</option>
          </optgroup>
          <optgroup label="Ensemble Spread & Chaos">
            <option value="total">Total Super Spread (σ_total)</option>
            <option value="intra_pooled">Pooled Internal Chaos (σ_intra)</option>
            <option value="gefs">GEFS Spread</option>
            <option value="aigefs">AIGEFS Spread</option>
            <option value="ifs">ECMWF IFS Spread</option>
            <option value="aifs">AIFS Spread</option>
          </optgroup>
        </select>

        <div class="tooltip-container">
          <button class="info-btn" id="infoBtn" onclick="toggleTooltip(event)" title="Click to view explanation of this underlay">ℹ</button>
          <div class="tooltip-card" id="tooltipCard">
            <div class="tooltip-title" id="ttTitle">Diagnostic Info</div>
            <div id="ttDesc">Description</div>
            <div class="tooltip-section"><b>Color Guide:</b> <span id="ttColors">Colors</span></div>
            <div class="tooltip-section"><b>Desk Action:</b> <span id="ttAction">Action</span></div>
          </div>
        </div>

        <div class="spread-colorbar" id="cbarBox">
          <span id="cbarMin">10m</span>
          <div class="colorbar-gradient" id="cbarGrad"></div>
          <span id="cbarMax">120m+</span>
        </div>
      </div>
    </div>

    <!-- Inline 1-Line Underlay Summary -->
    <div class="underlay-desc-bar" id="underlayDescBar">
      Clean basemap with interactive vector contours on top.
    </div>

    <!-- Row 2: Lead Scrubbing, Display Mode, and Consensus Modal Trigger -->
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

      <div style="display: flex; gap: 8px; align-items: center;">
        <button class="btn" style="background: #2563eb;" onclick="openConsensusModal()" title="View lead-time consensus tracking charts for all models">📈 Consensus Tracker</button>
      </div>
    </div>

    <!-- Row 3: Structured Model Toggles & Ghost Trails -->
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
        
        <span style="border-left: 1px solid var(--border); padding-left: 10px; margin-left: 4px;"></span>
        <label class="checkbox-item" title="Draws the Superensemble Mean from the previous 3 runs for the identical valid time.">
          <input type="checkbox" id="chk_ghost" onchange="draw()"> 
          <span class="swatch" style="background:#64748b; border: 1px dashed #fff;"></span> 
          <b>Past 3 Runs Ghost Trails (dprog/dt)</b>
        </label>
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

  <!-- Consensus Tracker Modal Dialogue -->
  <div id="consensusModal" class="modal-overlay" onclick="closeConsensusModal(event)">
    <div class="modal-content" onclick="event.stopPropagation()">
      <div class="modal-header">
        <h3>500 hPa Consensus Tracking (Model vs. Super Mean)</h3>
        <button class="modal-close" onclick="closeConsensusModal()">&times;</button>
      </div>
      <div class="modal-controls">
        <div style="display: flex; gap: 14px; align-items: center; font-size: 13px;">
          <b>Metric:</b>
          <label style="cursor: pointer;"><input type="radio" name="metricType" value="r" checked onchange="renderConsensusChart()"> Centered Pattern Correlation (r)</label>
          <label style="cursor: pointer;"><input type="radio" name="metricType" value="rmsd" onchange="renderConsensusChart()"> Spatial RMS Difference (meters)</label>
        </div>
        <div style="font-size: 11px; color: var(--text-muted);">
          Evaluates model wave phase alignment and amplitude bias against the 164-member Super Mean.
        </div>
      </div>
      <div class="chart-wrapper">
        <canvas id="consensusChart" width="850" height="380"></canvas>
      </div>
      <div class="modal-legend" id="modalLegend"></div>
    </div>
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
    let consensusDataCache = {{}};
    let currentStepData = null;
    let manifestData = {{ runs: [] }};

    const UNDERLAY_INFO = {{
      'none': {{
        title: 'Basemap Only (Clean)',
        desc: 'Standard Lambert Conformal basemap with state and coastline features. Leaves the canvas clean for spaghetti contour analysis.',
        colors: 'Neutral terrain styling.',
        action: 'Ideal for unencumbered inspection of tight contour clusters or manual wave tracking.',
        summary: 'Clean basemap with interactive vector contours on top.'
      }},
      'outlier_ens': {{
        title: 'Ensemble Mean Outliers (3-vs-1 Isolation)',
        desc: 'Flags when a single ensemble mean breaks away from the other three systems, accounting for ≥60% of total inter-model variance with a departure of ≥25m.',
        colors: 'Shaded by the rogue model color: Blue (GEFS), Purple (AIGEFS), Red (IFS), Green (AIFS).',
        action: 'Treat the flagged model with skepticism if the other 3 agree. If blank, models are either clustered or locked in a 2-vs-2 split.',
        summary: 'Highlights lone-wolf ensemble means that deviate sharply from the other 3 systems.'
      }},
      'outlier_det': {{
        title: 'Deterministic Outliers (|Z| ≥ 2.0σ)',
        desc: 'Identifies operational deterministic runs that sit in the extreme tails (≥2 standard deviations) relative to the 164-member superensemble spread.',
        colors: 'Dominant outlier gets solid color fill (Navy: GFS, Purple: AIGFS, Red: IFS, Green: AIFS). Dashed contours indicate co-outliers.',
        action: 'Avoid anchoring your forecast to a single deterministic run when it is flagged as an outlier.',
        summary: 'Highlights operational deterministic runs falling outside the 164-member ensemble envelope.'
      }},
      'inter': {{
        title: 'Inter-Model Disagreement (σ_inter)',
        desc: 'Standard deviation between the 4 ensemble means (GEFS, AIGEFS, IFS, AIFS). Isolates structural synoptic differences between modeling centers.',
        colors: 'Yellow to Dark Red (10m to 120m+ spread among ensemble means).',
        action: 'High values flag a "Model War" or synoptic bifurcation. Do not blend or trust the Super Mean here.',
        summary: 'Measures disagreement among the 4 ensemble means. High bullseyes indicate a Model War.'
      }},
      'physics_vs_ai': {{
        title: 'Physics minus AI (Phys - AI)',
        desc: 'Difference between the pooled traditional physics mean (GEFS + IFS) and the pooled AI emulator mean (AIGEFS + AIFS).',
        colors: 'Red (+10 to +80m): Physics higher / AI deeper. Blue (-10 to -80m): AI higher / Physics deeper.',
        action: 'Reveals whether differences are driven by AI smoothing of PV anomalies or physics convective feedbacks.',
        summary: 'Compares Physics ensembles (GEFS+IFS) against Deep-Learning AI ensembles (AIGEFS+AIFS).'
      }},
      'monotonic_trend': {{
        title: '4-Run Monotonic Trend (Deepening vs Building)',
        desc: 'Measures persistent, directional height changes across the last 4 consecutive runs (0h, -12h, -24h, -36h) for this identical valid time.',
        colors: 'Blue (-20 to -80m): Steady multi-run deepening trough. Red (+20 to +80m): Steady multi-run building ridge.',
        action: 'Confirms a genuine synoptic trend across shifts rather than single-run model noise.',
        summary: 'Shows where models have consistently trended deeper (blue) or stronger (red) over 4 cycles.'
      }},
      'flipflop': {{
        title: '4-Run Flip-Flop Volatility (Oscillation Energy)',
        desc: 'Calculates path-length hunting: identifies regions where guidance has bounced back and forth across consecutive shifts without settling on a solution.',
        colors: 'Yellow to Red (10m to 80m+ of oscillating volatility).',
        action: 'Do not chase the latest run inside a bright bullseye; guidance is unstable.',
        summary: 'Warns of "washing machine" zones where models are alternating between deeper and weaker.'
      }},
      'convergence': {{
        title: 'Model Convergence Rate (Δσ_inter)',
        desc: 'Compares inter-model disagreement now versus 12 hours ago for this valid time.',
        colors: 'Green (-30m): Models converging into consensus. Purple (+30m): Models diverging further apart.',
        action: 'Green indicates growing confidence. Purple flags an emerging synoptic bifurcation.',
        summary: 'Green shows models locking onto consensus; Purple flags an emerging model war.'
      }},
      'total': {{
        title: 'Total Superensemble Spread (σ_total)',
        desc: 'Standard deviation across all 164 individual ensemble members pooled together. Measures the total atmospheric uncertainty envelope.',
        colors: 'Yellow to Dark Red (10m to 120m+ spread across all members).',
        action: 'Represents the big-picture confidence boundary for synoptic feature placement.',
        summary: 'Overall uncertainty envelope across all 164 members pooled together.'
      }},
      'intra_pooled': {{
        title: 'Pooled Internal Chaos (σ_intra)',
        desc: 'Average standard deviation occurring internally inside each ensemble system. Measures sensitivity to initial conditions and chaotic dispersion.',
        colors: 'Yellow to Dark Red (10m to 120m+ internal chaotic spread).',
        action: 'High values mean internal wave timing or shortwave dispersion is chaotic.',
        summary: 'Average chaotic disagreement within each individual ensemble system.'
      }},
      'gefs': {{
        title: 'GEFS Spread',
        desc: 'Internal standard deviation among the 31 physics-based GEFS members.',
        colors: 'Yellow to Dark Red.',
        action: 'Evaluates NCEP physics suite sensitivity.',
        summary: 'Internal spread among the 31 NCEP GEFS members.'
      }},
      'aigefs': {{
        title: 'AIGEFS Spread',
        desc: 'Internal standard deviation among the 31 NOAA AI-GEFS members.',
        colors: 'Yellow to Dark Red.',
        action: 'Evaluates NCEP AI suite dispersion.',
        summary: 'Internal spread among the 31 NOAA AIGEFS deep-learning members.'
      }},
      'ifs': {{
        title: 'ECMWF IFS Spread',
        desc: 'Internal standard deviation among the 51 ECMWF IFS members.',
        colors: 'Yellow to Dark Red.',
        action: 'Evaluates European physics suite dispersion.',
        summary: 'Internal spread among the 51 ECMWF IFS members.'
      }},
      'aifs': {{
        title: 'AIFS Spread',
        desc: 'Internal standard deviation among the 51 ECMWF AIFS members.',
        colors: 'Yellow to Dark Red.',
        action: 'Evaluates European AI suite dispersion.',
        summary: 'Internal spread among the 51 ECMWF AIFS deep-learning members.'
      }}
    }};

    function toggleTooltip(e) {{
      e.stopPropagation();
      const card = document.getElementById('tooltipCard');
      card.classList.toggle('show');
    }}

    document.addEventListener('click', (e) => {{
      const container = document.querySelector('.tooltip-container');
      if (container && !container.contains(e.target)) {{
        document.getElementById('tooltipCard').classList.remove('show');
      }}
    }});

    function parseUtcDate(str) {{
      if (!str) return NaN;
      const clean = str.replace(/[Zz]/g, '').trim();
      const parts = clean.split(' ');
      if (parts.length !== 2) return NaN;
      const [d, t] = parts;
      const [yr, mo, da] = d.split('-').map(Number);
      const [hr, mi] = t.split(':').map(Number);
      return Date.UTC(yr, mo - 1, da, hr, mi, 0);
    }}

    function parseRunId(id) {{
      if (!id || id.length < 11) return NaN;
      const yr = parseInt(id.slice(0, 4), 10);
      const mo = parseInt(id.slice(4, 6), 10) - 1;
      const da = parseInt(id.slice(6, 8), 10);
      const hr = parseInt(id.slice(9, 11), 10);
      return Date.UTC(yr, mo, da, hr, 0, 0);
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

    function updateBackground() {{
      const bg = document.getElementById('bgSelect').value;
      const fxx = document.getElementById('leadSelect').value;
      const runId = document.getElementById('runSelect').value;
      const img = document.getElementById('mapUnderlay');
      const cbar = document.getElementById('cbarBox');
      const cbarGrad = document.getElementById('cbarGrad');
      const cbarMin = document.getElementById('cbarMin');
      const cbarMax = document.getElementById('cbarMax');

      const info = UNDERLAY_INFO[bg] || UNDERLAY_INFO['none'];
      document.getElementById('ttTitle').textContent = info.title;
      document.getElementById('ttDesc').textContent = info.desc;
      document.getElementById('ttColors').textContent = info.colors;
      document.getElementById('ttAction').textContent = info.action;
      document.getElementById('underlayDescBar').textContent = info.summary;

      img.onerror = function() {{
        img.onerror = null;
        img.src = 'basemap.png';
        cbar.style.display = 'none';
      }};

      if (bg === 'none') {{
        img.src = 'basemap.png';
        cbar.style.display = 'none';
      }} else if (bg === 'outlier_det') {{
        img.src = `runs/${{runId}}/spread/outlier_det_f${{fxx}}.png`;
        cbar.style.display = 'flex';
        cbarGrad.style.background = 'linear-gradient(to right, #001155 0% 25%, #4a235a 25% 50%, #78281f 50% 75%, #145a32 75% 100%)';
        cbarMin.textContent = 'GFS/AIGFS';
        cbarMax.textContent = 'IFS/AIFS (≥2σ)';
      }} else if (bg === 'outlier_ens') {{
        img.src = `runs/${{runId}}/spread/outlier_ens_f${{fxx}}.png`;
        cbar.style.display = 'flex';
        cbarGrad.style.background = 'linear-gradient(to right, #0044ff 0% 25%, #8e44ad 25% 50%, #d63031 50% 75%, #00b894 75% 100%)';
        cbarMin.textContent = 'GEFS/AIGEFS';
        cbarMax.textContent = 'IFS/AIFS (≥25m)';
      }} else if (bg === 'physics_vs_ai' || bg === 'monotonic_trend') {{
        img.src = `runs/${{runId}}/spread/${{bg}}_f${{fxx}}.png`;
        cbar.style.display = 'flex';
        cbarGrad.style.background = 'linear-gradient(to right, #2166ac, #67a9cf, #d1e5f0, #f7f7f7, #fddbc7, #ef8a62, #b2182b)';
        cbarMin.textContent = bg === 'monotonic_trend' ? '-80m (Deep)' : '-80m (AI)';
        cbarMax.textContent = bg === 'monotonic_trend' ? '+80m (Build)' : '+80m (Phys)';
      }} else if (bg === 'convergence') {{
        img.src = `runs/${{runId}}/spread/convergence_f${{fxx}}.png`;
        cbar.style.display = 'flex';
        cbarGrad.style.background = 'linear-gradient(to right, #27ae60, #a9dfbf, #f7f7f7, #d7bde2, #8e44ad)';
        cbarMin.textContent = '-30m (Agree)';
        cbarMax.textContent = '+30m (Dis)';
      }} else if (bg === 'flipflop') {{
        img.src = `runs/${{runId}}/spread/flipflop_f${{fxx}}.png`;
        cbar.style.display = 'flex';
        cbarGrad.style.background = 'linear-gradient(to right, #ffffcc, #fed976, #feb24c, #fd8d3c, #f03b20, #bd0026)';
        cbarMin.textContent = '10m';
        cbarMax.textContent = '80m+ (Flip)';
      }} else {{
        img.src = `runs/${{runId}}/spread/${{bg}}_f${{fxx}}.png`;
        cbar.style.display = 'flex';
        cbarGrad.style.background = 'linear-gradient(to right, #ffffcc, #fed976, #feb24c, #fd8d3c, #f03b20, #bd0026)';
        cbarMin.textContent = '10m';
        cbarMax.textContent = '120m+';
      }}
    }}

    function updateDprogBadge() {{
      const isSync = document.getElementById('chk_dprogdt').checked;
      const statusEl = document.getElementById('dprogStatus');
      if (!statusEl) return;
      if (isSync && currentStepData && currentStepData.valid) {{
        statusEl.innerHTML = `(Locked: <b>${{currentStepData.valid}}</b>)`;
      }} else {{
        statusEl.innerHTML = `(Lead-locked)`;
      }}
    }}

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
      if (syncDprog && currentStepData && currentStepData.valid) {{
        const targetValidMs = parseUtcDate(currentStepData.valid);
        const newRunId = document.getElementById('runSelect').value;
        const newRunInitMs = parseRunId(newRunId);

        if (!isNaN(targetValidMs) && !isNaN(newRunInitMs)) {{
          const diffHours = Math.round((targetValidMs - newRunInitMs) / 3600000);
          if (FXX_STEPS.includes(diffHours)) {{
            document.getElementById('leadSelect').value = String(diffHours).padStart(3, '0');
          }} else {{
            const clamped = Math.max(0, Math.min(240, Math.round(diffHours / 12) * 12));
            document.getElementById('leadSelect').value = String(clamped).padStart(3, '0');
          }}
        }}
      }}
      loadStepData();
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowLeft') {{ e.preventDefault(); stepLead(-1); }}
      if (e.key === 'ArrowRight') {{ e.preventDefault(); stepLead(1); }}
      if (e.key === 'ArrowUp') {{ e.preventDefault(); stepRun(-1); }}
      if (e.key === 'ArrowDown') {{ e.preventDefault(); stepRun(1); }}
    }});

    // Helper to format live consensus correlation & rmsd for legend
    function formatConsensusMeta(m) {{
      if (!m || m.r === null || m.r === undefined) return '';
      const rStr = m.r >= 0 ? `r=${{m.r.toFixed(2)}}` : `r=${{m.r.toFixed(2)}}`;
      return ` <span style="font-size:10px; color:#475569; font-weight:normal;">(${{rStr}}, ±${{Math.round(m.rmsd)}}m)</span>`;
    }}

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
      const showGhost = document.getElementById('chk_ghost').checked;

      document.getElementById('mapTitleStamp').innerHTML = 
        `500 hPa Height (m) | Lead: F${{String(currentStepData.fxx).padStart(3, '0')}}<br>` +
        `Init: ${{currentStepData.init}}Z | Valid: ${{currentStepData.valid}}`;

      updateDprogBadge();

      const legendItems = [];
      const m = currentStepData.metrics || {{}};

      function drawLines(layerDict, color, width, alpha, dashArray = []) {{
        if (!layerDict) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.globalAlpha = alpha;
        ctx.setLineDash(dashArray);

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
          currentStepData.gefs_members.forEach(mem => drawLines(mem, '#2980b9', 0.9, 0.55));
          legendItems.push({{ label: `GEFS Members (${{currentStepData.gefs_members.length}})`, color: '#2980b9', dash: false }});
        }}
        if (showAigefs && currentStepData.aigefs_members) {{
          currentStepData.aigefs_members.forEach(mem => drawLines(mem, '#9b59b6', 0.9, 0.55));
          legendItems.push({{ label: `AIGEFS Members (${{currentStepData.aigefs_members.length}})`, color: '#9b59b6', dash: false }});
        }}
        if (showIfs && currentStepData.ifs_members) {{
          currentStepData.ifs_members.forEach(mem => drawLines(mem, '#e74c3c', 0.9, 0.55));
          legendItems.push({{ label: `IFS Members (${{currentStepData.ifs_members.length}})`, color: '#e74c3c', dash: false }});
        }}
        if (showAifs && currentStepData.aifs_members) {{
          currentStepData.aifs_members.forEach(mem => drawLines(mem, '#00b894', 0.9, 0.55));
          legendItems.push({{ label: `AIFS Members (${{currentStepData.aifs_members.length}})`, color: '#00b894', dash: false }});
        }}
      }}

      // 2. Deterministics (Dashed with Live Consensus Metrics)
      if (showGfs && currentStepData.gfs_det) {{
        drawLines(currentStepData.gfs_det, '#001155', 2.0, 1.0, [6, 4]);
        legendItems.push({{ label: `GFS Det${{formatConsensusMeta(m.gfs_det)}}`, color: '#001155', dash: true }});
      }}
      if (showAigfs && currentStepData.aigfs_det) {{
        drawLines(currentStepData.aigfs_det, '#4a235a', 2.0, 1.0, [6, 4]);
        legendItems.push({{ label: `AIGFS Det${{formatConsensusMeta(m.aigfs_det)}}`, color: '#4a235a', dash: true }});
      }}
      if (showEcmwf && currentStepData.ecmwf_det) {{
        drawLines(currentStepData.ecmwf_det, '#78281f', 2.0, 1.0, [6, 4]);
        legendItems.push({{ label: `IFS Det${{formatConsensusMeta(m.ecmwf_det)}}`, color: '#78281f', dash: true }});
      }}
      if (showAifsDet && currentStepData.aifs_det) {{
        drawLines(currentStepData.aifs_det, '#145a32', 2.0, 1.0, [6, 4]);
        legendItems.push({{ label: `AIFS Det${{formatConsensusMeta(m.aifs_det)}}`, color: '#145a32', dash: true }});
      }}

      // 3. Ensemble Means (with Live Consensus Metrics)
      if (showGefs && currentStepData.gefs_mean) {{
        drawLines(currentStepData.gefs_mean, '#0044ff', 2.4, 1.0);
        legendItems.push({{ label: `GEFS Mean${{formatConsensusMeta(m.gefs_mean)}}`, color: '#0044ff', dash: false }});
      }}
      if (showAigefs && currentStepData.aigefs_mean) {{
        drawLines(currentStepData.aigefs_mean, '#8e44ad', 2.4, 1.0);
        legendItems.push({{ label: `AIGEFS Mean${{formatConsensusMeta(m.aigefs_mean)}}`, color: '#8e44ad', dash: false }});
      }}
      if (showIfs && currentStepData.ifs_mean) {{
        drawLines(currentStepData.ifs_mean, '#d63031', 2.4, 1.0);
        legendItems.push({{ label: `IFS Mean${{formatConsensusMeta(m.ifs_mean)}}`, color: '#d63031', dash: false }});
      }}
      if (showAifs && currentStepData.aifs_mean) {{
        drawLines(currentStepData.aifs_mean, '#00b894', 2.4, 1.0);
        legendItems.push({{ label: `AIFS Mean${{formatConsensusMeta(m.aifs_mean)}}`, color: '#00b894', dash: false }});
      }}

      // 4. Past 3 Runs Ghost Trails (dprog/dt overlay)
      if (showGhost && currentStepData && currentStepData.valid) {{
        const curIdx = manifestData.runs.findIndex(r => r.id === document.getElementById('runSelect').value);
        if (curIdx !== -1) {{
          const ghostConfigs = [
            {{ offset: 1, label: '-12h Run', color: '#334155', width: 2.0, dash: [6, 4], alpha: 0.85 }},
            {{ offset: 2, label: '-24h Run', color: '#64748b', width: 1.6, dash: [4, 4], alpha: 0.70 }},
            {{ offset: 3, label: '-36h Run', color: '#94a3b8', width: 1.3, dash: [2, 3], alpha: 0.55 }}
          ];

          ghostConfigs.forEach(cfg => {{
            const pastRunIdx = curIdx + cfg.offset;
            if (pastRunIdx < manifestData.runs.length) {{
              const pastRunObj = manifestData.runs[pastRunIdx];
              const targetValidMs = parseUtcDate(currentStepData.valid);
              const pastInitMs = parseRunId(pastRunObj.id);
              const pastFxx = Math.round((targetValidMs - pastInitMs) / 3600000);

              if (FXX_STEPS.includes(pastFxx)) {{
                const fxxStr = String(pastFxx).padStart(3, '0');
                const pastCacheKey = `${{pastRunObj.id}}_${{fxxStr}}`;
                
                if (dataCache[pastCacheKey]) {{
                  const pastData = dataCache[pastCacheKey];
                  if (pastData.super_mean) {{
                    drawLines(pastData.super_mean, cfg.color, cfg.width, cfg.alpha, cfg.dash);
                    legendItems.push({{ label: `Super Mean (${{cfg.label}})`, color: cfg.color, dash: true }});
                  }}
                }} else {{
                  fetch(`runs/${{pastRunObj.id}}/data/f${{fxxStr}}.json`)
                    .then(r => r.json())
                    .then(json => {{
                      dataCache[pastCacheKey] = json;
                      draw();
                    }}).catch(() => {{}});
                }}
              }}
            }}
          }});
        }}
      }}

      // 5. Superensemble Mean
      if (showSuper && currentStepData.super_mean) {{
        drawLines(currentStepData.super_mean, '#000000', 3.2, 1.0);
        legendItems.push({{ label: 'Superensemble Mean (Consensus)', color: '#000000', dash: false }});

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

      // Dynamic Legend Box
      const legendBox = document.getElementById('legendBox');
      legendBox.innerHTML = legendItems.map(item => `
        <div class="legend-line">
          <span class="line-sample" style="background:${{item.color}}; border-bottom: ${{item.dash ? '2px dashed #000' : 'none'}};"></span>
          <span>${{item.label}}</span>
        </div>
      `).join('');
    }}

    // =========================================================================
    // 10. CONSENSUS TRACKER MODAL & CANVAS CHART RENDERER
    // =========================================================================
    const MODEL_STYLES = {{
      'gefs_mean':   {{ label: 'GEFS Mean', color: '#0044ff', dash: [] }},
      'aigefs_mean': {{ label: 'AIGEFS Mean', color: '#8e44ad', dash: [] }},
      'ifs_mean':    {{ label: 'IFS Mean', color: '#d63031', dash: [] }},
      'aifs_mean':   {{ label: 'AIFS Mean', color: '#00b894', dash: [] }},
      'gfs_det':     {{ label: 'GFS Det', color: '#001155', dash: [5, 3] }},
      'aigfs_det':   {{ label: 'AIGFS Det', color: '#4a235a', dash: [5, 3] }},
      'ecmwf_det':   {{ label: 'IFS Det', color: '#78281f', dash: [5, 3] }},
      'aifs_det':    {{ label: 'AIFS Det', color: '#145a32', dash: [5, 3] }}
    }};

    async function openConsensusModal() {{
      const modal = document.getElementById('consensusModal');
      modal.classList.add('open');
      const runId = document.getElementById('runSelect').value;

      if (!consensusDataCache[runId]) {{
        try {{
          const resp = await fetch(`runs/${{runId}}/consensus_metrics.json`);
          consensusDataCache[runId] = await resp.json();
        }} catch (e) {{
          console.warn("Could not load consensus_metrics.json for run", runId);
        }}
      }}
      renderConsensusChart();
    }}

    function closeConsensusModal(e) {{
      if (e && e.target !== e.currentTarget && !e.target.classList.contains('modal-close')) return;
      document.getElementById('consensusModal').classList.remove('open');
    }}

    function renderConsensusChart() {{
      const runId = document.getElementById('runSelect').value;
      const dataObj = consensusDataCache[runId];
      const canvas = document.getElementById('consensusChart');
      const ctx = canvas.getContext('2d');
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const metricType = document.querySelector('input[name="metricType"]:checked').value;
      const currentLead = parseInt(document.getElementById('leadSelect').value, 10);

      // Margins
      const padL = 60, padR = 30, padT = 30, padB = 45;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      // Coordinate scaling
      const xMin = 0, xMax = 240;
      let yMin = 0.60, yMax = 1.00;
      if (metricType === 'rmsd') {{
        yMin = 0;
        yMax = 90;
      }}

      function getX(f) {{ return padL + ((f - xMin) / (xMax - xMin)) * plotW; }}
      function getY(v) {{ return padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH; }}

      // 1. Gridlines & Axes
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1.0;
      ctx.setLineDash([]);
      ctx.font = '11px sans-serif';
      ctx.fillStyle = '#94a3b8';

      // X Gridlines (every 24 hours)
      for (let f = 0; f <= 240; f += 24) {{
        const x = getX(f);
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
        ctx.textAlign = 'center';
        ctx.fillText(`F${{String(f).padStart(3, '0')}}`, x, H - padB + 16);
      }}

      // Y Gridlines
      if (metricType === 'r') {{
        const yTicks = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00];
        yTicks.forEach(val => {{
          const y = getY(val);
          ctx.beginPath();
          ctx.moveTo(padL, y);
          ctx.lineTo(W - padR, y);
          ctx.stroke();
          ctx.textAlign = 'right';
          ctx.fillText(val.toFixed(2), padL - 8, y + 4);
        }});
      }} else {{
        const yTicks = [0, 20, 40, 60, 80];
        yTicks.forEach(val => {{
          const y = getY(val);
          ctx.beginPath();
          ctx.moveTo(padL, y);
          ctx.lineTo(W - padR, y);
          ctx.stroke();
          ctx.textAlign = 'right';
          ctx.fillText(`${{val}}m`, padL - 8, y + 4);
        }});
      }}

      // Axis Titles
      ctx.save();
      ctx.translate(16, padT + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = '#cbd5e1';
      ctx.fillText(metricType === 'r' ? 'Centered Pattern Correlation (r)' : 'Spatial RMS Difference (meters)', 0, 0);
      ctx.restore();

      // Current Lead Indicator Line
      const curX = getX(currentLead);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(curX, padT);
      ctx.lineTo(curX, padT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#f59e0b';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Active F${{String(currentLead).padStart(3, '0')}}`, curX, padT - 8);

      if (!dataObj || !dataObj.metrics) return;

      // 2. Plot Model Curves
      const metricsData = dataObj.metrics;
      const modelKeys = Object.keys(MODEL_STYLES);

      modelKeys.forEach(mKey => {{
        const style = MODEL_STYLES[mKey];
        ctx.strokeStyle = style.color;
        ctx.lineWidth = style.dash.length ? 1.8 : 2.4;
        ctx.setLineDash(style.dash);
        ctx.beginPath();
        let started = false;

        FXX_STEPS.forEach(f => {{
          const fStr = String(f).padStart(3, '0');
          const stepMeta = metricsData[fStr];
          if (stepMeta && stepMeta[mKey]) {{
            const val = stepMeta[mKey][metricType];
            if (val !== null && val !== undefined) {{
              const clampedVal = Math.max(yMin, Math.min(yMax, val));
              const x = getX(f);
              const y = getY(clampedVal);
              if (!started) {{
                ctx.moveTo(x, y);
                started = true;
              }} else {{
                ctx.lineTo(x, y);
              }}
            }}
          }}
        }});
        ctx.stroke();
      }});
      ctx.setLineDash([]);

      // 3. Render Modal Legend
      const legEl = document.getElementById('modalLegend');
      legEl.innerHTML = modelKeys.map(k => {{
        const s = MODEL_STYLES[k];
        return `
          <div style="display:flex; align-items:center; gap:6px;">
            <span style="width:18px; height:3px; background:${{s.color}}; display:inline-block; border-bottom:${{s.dash.length ? '2px dashed #000' : 'none'}};"></span>
            <span>${{s.label}}</span>
          </div>
        `;
      }}).join('');
    }}

    async function initViewer() {{
      try {{
        const resp = await fetch('manifest.json?t=' + Date.now());
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
