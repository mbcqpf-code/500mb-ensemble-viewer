import os
import shutil
import warnings
import concurrent.futures
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Headless backend for GitHub Actions
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie

warnings.filterwarnings('ignore')

# 1. DIRECTORY SETUP
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

# 2. RUN CADENCE & TIMING
FXX_STEPS = list(range(0, 241, 12))  # 0 to 240 hours every 12 hours
ALL_CONTOURS = list(range(4980, 6001, 60))

now = datetime.utcnow()
if (now.hour == 1 and now.minute > 30) or (2 <= now.hour <= 12) or (now.hour == 13 and now.minute <= 30):
    base_date = now - timedelta(days=1)
    global_cycle = "12"
else:
    base_date = now - timedelta(days=1) if now.hour <= 1 else now
    global_cycle = "00"

global_init_time = base_date.strftime(f'%Y-%m-%d {global_cycle}:00')
print(f"🚀 Processing Global Cycle: {global_init_time}")

# Domain definition
lon_min, lon_max = -130.0, -60.0
lat_min, lat_max = 20.0, 55.0
grid_lats = np.arange(lat_min, lat_max + 0.5, 0.5)
grid_lons = np.arange(lon_min, lon_max + 0.5, 0.5)
target_lons, target_lats = np.meshgrid(grid_lons, grid_lats)

def standardize_grid(ds):
    if ds is None: return None
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

# 3. ENSEMBLE FETCHERS
def fetch_gefs_lead(fxx):
    members = ["c00"] + [f"p{str(i).zfill(2)}" for i in range(1, 31)]
    def _worker(m):
        try:
            H = Herbie(global_init_time, model="gefs", product="atmos.5", member=m, fxx=fxx)
            return standardize_grid(H.xarray(":HGT:500 mb:"))
        except Exception:
            return None
    grids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for res in executor.map(_worker, members):
            if res is not None and res.ndim == 2: grids.append(res)
    return grids

def fetch_ifs_ensemble(fxx):
    grids = []
    try:
        H = Herbie(global_init_time, model="ifs", product="enfo", fxx=fxx)
        ds_list = H.xarray(":gh:500")
        if not isinstance(ds_list, list): ds_list = [ds_list]
        for ds in ds_list:
            g = standardize_grid(ds)
            if g is not None:
                if g.ndim == 3:
                    for i in range(g.shape[0]): grids.append(g[i])
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
                if not isinstance(ds_list, list): ds_list = [ds_list]
                for ds in ds_list:
                    var_key = list(ds.data_vars)[0]
                    da = ds[var_key].squeeze()
                    dim_names = list(da.dims)
                    if any(d in dim_names for d in ['number', 'realization']):
                        m_dim = 'number' if 'number' in dim_names else 'realization'
                        for m_idx in range(len(da[m_dim])):
                            g = standardize_grid(da.isel({m_dim: m_idx}).to_dataset(name=var_key))
                            if g is not None and g.ndim == 2: grids.append(g)
                    else:
                        g = standardize_grid(da.to_dataset(name=var_key))
                        if g is not None and g.ndim == 2: grids.append(g)
            except Exception:
                continue
    except Exception:
        pass
    return grids

def fetch_det(model, product, search_str, fxx):
    try:
        H = Herbie(global_init_time, model=model, product=product, fxx=fxx)
        ds = H.xarray(search_str)
        if isinstance(ds, list): ds = ds[0]
        g = standardize_grid(ds)
        if g is not None and g.ndim == 2: return g
    except Exception:
        pass
    return None

def fetch_aigfs(fxx):
    for sq in [":HGT:500 mb:", ":HGT:500:"]:
        try:
            H = Herbie(global_init_time, model="aigfs", product="pres", fxx=fxx)
            ds = H.xarray(sq)
            if isinstance(ds, list): ds = ds[0]
            g = standardize_grid(ds)
            if g is not None and g.ndim == 2: return g
        except Exception:
            pass
    return None

# 4. RENDER FRAME ROUTINE
def render_frame(data, fxx, mode, filename):
    fig = plt.figure(figsize=(14, 9), dpi=120)
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0))
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor="#fdfefe")
    ax.add_feature(cfeature.OCEAN, facecolor="#eef2f5")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.1, edgecolor="#222222")
    ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="#222222")
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#777777", linestyle=":")
    
    levels = ALL_CONTOURS
    handles, labels = [], []
    plot_members = (mode == "spaghetti")

    # GEFS
    if plot_members and data['gefs_mems']:
        for m in data['gefs_mems']:
            ax.contour(target_lons, target_lats, m, levels=levels, colors='#5b9bd5', linewidths=0.5, alpha=0.3, transform=ccrs.PlateCarree())
        handles.append(mlines.Line2D([], [], color='#5b9bd5', lw=1.0, alpha=0.6))
        labels.append(f"GEFS Members ({len(data['gefs_mems'])})")
    if data['gefs_mean'] is not None:
        ax.contour(target_lons, target_lats, data['gefs_mean'], levels=levels, colors='#0033cc', linewidths=2.2, transform=ccrs.PlateCarree())
        handles.append(mlines.Line2D([], [], color='#0033cc', lw=2.2))
        labels.append("GEFS Mean")

    # IFS
    if plot_members and data['ifs_mems']:
        for m in data['ifs_mems']:
            ax.contour(target_lons, target_lats, m, levels=levels, colors='#f1948a', linewidths=0.5, alpha=0.3, transform=ccrs.PlateCarree())
        handles.append(mlines.Line2D([], [], color='#f1948a', lw=1.0, alpha=0.6))
        labels.append(f"IFS Members ({len(data['ifs_mems'])})")
    if data['ifs_mean'] is not None:
        ax.contour(target_lons, target_lats, data['ifs_mean'], levels=levels, colors='#c0392b', linewidths=2.2, transform=ccrs.PlateCarree())
        handles.append(mlines.Line2D([], [], color='#c0392b', lw=2.2))
        labels.append("IFS Mean")

    # AIFS
    if plot_members and data['aifs_mems']:
        for m in data['aifs_mems']:
            ax.contour(target_lons, target_lats, m, levels=levels, colors='#82e0aa', linewidths=0.5, alpha=0.3, transform=ccrs.PlateCarree())
        handles.append(mlines.Line2D([], [], color='#82e0aa', lw=1.0, alpha=0.6))
        labels.append(f"AIFS Members ({len(data['aifs_mems'])})")
    if data['aifs_mean'] is not None:
        ax.contour(target_lons, target_lats, data['aifs_mean'], levels=levels, colors='#196f3d', linewidths=2.2, transform=ccrs.PlateCarree())
        handles.append(mlines.Line2D([], [], color='#196f3d', lw=2.2))
        labels.append("AIFS Mean")

    # Deterministics
    if data['det_gfs'] is not None:
        ax.contour(target_lons, target_lats, data['det_gfs'], levels=levels, colors='#001155', linewidths=1.7, linestyles='--', transform=ccrs.PlateCarree())
        handles.append(mlines.Line2D([], [], color='#001155', lw=1.7, ls='--'))
        labels.append("GFS Det")
    if data['det_aigfs'] is not None:
        ax.contour(target_lons, target_lats, data['det_aigfs'], levels=levels, colors='#8e44ad', linewidths=1.7, linestyles='--', transform=ccrs.PlateCarree())
        handles.append(mlines.Line2D([], [], color='#8e44ad', lw=1.7, ls='--'))
        labels.append("AIGFS Det")

    # Superensemble Mean
    if data['super_mean'] is not None:
        cs = ax.contour(target_lons, target_lats, data['super_mean'], levels=levels, colors='#000000', linewidths=3.0, transform=ccrs.PlateCarree())
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d m', colors='black')
        handles.append(mlines.Line2D([], [], color='#000000', lw=3.0))
        labels.append("Superensemble Mean")

    valid_time = datetime.strptime(global_init_time, '%Y-%m-%d %H:%M') + timedelta(hours=fxx)
    ax.set_title(
        f"500 hPa Height (4980-6000m every 60m) | Forecast Lead: F{fxx:03d}\n"
        f"Init: {global_init_time}Z | Valid: {valid_time.strftime('%Y-%m-%d %H:%MZ')}",
        fontsize=12, loc='left', weight='bold'
    )
    if handles:
        ax.legend(handles, labels, loc='upper right', framealpha=0.9, fontsize=9)
    plt.savefig(f"{OUTPUT_DIR}/images/{filename}", bbox_inches='tight', dpi=120)
    plt.close(fig)

# 5. PRODUCTION BATCH LOOP
for fxx in FXX_STEPS:
    print(f"📦 Processing Lead Time F{fxx:03d}...")
    gefs_mems = fetch_gefs_lead(fxx)
    ifs_mems  = fetch_ifs_ensemble(fxx)
    aifs_mems = fetch_aifs_ensemble(fxx)

    gefs_mean = np.mean(np.stack(gefs_mems), axis=0) if gefs_mems else None
    ifs_mean  = np.mean(np.stack(ifs_mems), axis=0) if ifs_mems else None
    aifs_mean = np.mean(np.stack(aifs_mems), axis=0) if aifs_mems else None
    
    all_m = gefs_mems + ifs_mems + aifs_mems
    super_mean = np.mean(np.stack(all_m), axis=0) if all_m else None
    
    step_data = {
        'gefs_mems': gefs_mems, 'ifs_mems': ifs_mems, 'aifs_mems': aifs_mems,
        'gefs_mean': gefs_mean, 'ifs_mean': ifs_mean, 'aifs_mean': aifs_mean,
        'super_mean': super_mean,
        'det_gfs': fetch_det("gfs", "pgrb2.0p25", ":HGT:500 mb:", fxx),
        'det_aigfs': fetch_aigfs(fxx)
    }

    # Render Spaghetti Frame and Means Only Frame
    render_frame(step_data, fxx, "spaghetti", f"spaghetti_f{fxx:03d}.png")
    render_frame(step_data, fxx, "means", f"means_f{fxx:03d}.png")

# 6. WRITE INDEX.HTML FOR GITHUB PAGES
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>500mb Multi-Ensemble Viewer</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #0f172a; color: #f8fafc; text-align: center; }}
    .header {{ padding: 16px 20px; background: #1e293b; border-bottom: 1px solid #334155; }}
    .controls {{ display: flex; justify-content: center; align-items: center; gap: 15px; margin: 15px auto; flex-wrap: wrap; max-width: 1000px; }}
    .btn {{ background: #3b82f6; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: bold; }}
    .btn:hover {{ background: #2563eb; }}
    select {{ padding: 8px 12px; border-radius: 6px; background: #334155; color: white; border: 1px solid #475569; font-size: 14px; }}
    .viewer-container {{ max-width: 1300px; margin: 0 auto; padding: 10px; }}
    img {{ width: 100%; max-width: 1200px; height: auto; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
    .footer {{ margin-top: 20px; padding: 12px; font-size: 13px; color: #94a3b8; }}
    kbd {{ background: #334155; padding: 2px 6px; border-radius: 4px; border: 1px solid #475569; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="header">
    <h2 style="margin: 0 0 5px 0;">500 hPa Multi-Ensemble & Deterministic Spaghetti Viewer</h2>
    <div style="font-size: 14px; color: #cbd5e1;">Baseline Cycle: <b>{global_init_time}Z</b> | Leads: F000 - F240 (12h Step)</div>
  </div>

  <div class="controls">
    <button class="btn" onclick="step(-1)">◀ Prev Step</button>
    <select id="leadSelect" onchange="updateImage()">
      {"".join([f'<option value="{f:03d}">Forecast Hour: F{f:03d} (+{f}h)</option>' for f in FXX_STEPS])}
    </select>
    <button class="btn" onclick="step(1)">Next Step ▶</button>
    
    <label style="margin-left: 20px; font-size: 14px;">
      <b>Layer:</b>
      <select id="modeSelect" onchange="updateImage()">
        <option value="spaghetti">All Members (Spaghetti) + Means</option>
        <option value="means">Ensemble Means Only</option>
      </select>
    </label>
  </div>

  <div class="viewer-container">
    <img id="displayImg" src="images/spaghetti_f000.png" alt="500mb Forecast">
  </div>

  <div class="footer">
    Keyboard Shortcuts: Use <kbd>←</kbd> and <kbd>→</kbd> arrow keys to step lead times. | Automated with GitHub Actions.
  </div>

  <script>
    const steps = {FXX_STEPS};
    let currentIndex = 0;

    function updateImage() {{
      const fxx = document.getElementById('leadSelect').value;
      const mode = document.getElementById('modeSelect').value;
      document.getElementById('displayImg').src = `images/${{mode}}_f${{fxx}}.png`;
    }}

    function step(delta) {{
      const select = document.getElementById('leadSelect');
      let newIdx = select.selectedIndex + delta;
      if (newIdx >= 0 && newIdx < select.options.length) {{
        select.selectedIndex = newIdx;
        updateImage();
      }}
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowLeft') step(-1);
      if (e.key === 'ArrowRight') step(1);
    }});
  </script>
</body>
</html>
"""

with open(f"{OUTPUT_DIR}/index.html", "w") as f:
    f.write(html_content)

print(f"✅ Run complete. Dashboard generated in '{OUTPUT_DIR}' directory.")
