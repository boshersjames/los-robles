#!/usr/bin/env python3
"""
coursekit: build the data for any GSPro / OPCD course from public USGS LiDAR, NAIP imagery and OpenStreetMap.

Each course lives in its own folder with a course.json. Run every step from inside that folder.

  python coursekit.py new  <name> <lat> <lon>     make ~/courses/<name>/course.json for a 2000 m square centred there
  python coursekit.py lidar  [--survey NAME]      download + clip USGS LiDAR      -> points.npz
  python coursekit.py satellite                   NAIP aerial photo                -> satellite.jpg
  python coursekit.py heightmap                   Unity terrain (2049 x 2049 RAW)  -> heightmap_2049.raw  (+ import settings)
  python coursekit.py trees  [--buffer 150]       trees from LiDAR                 -> trees.csv, trees_on_satellite.png
  python coursekit.py brush  [--buffer 200]       brush / chaparral from LiDAR     -> shrubs.csv, brush_mask.png, brush_on_satellite.png
  python coursekit.py osm                         what map.osm has + cart paths    -> cartpaths.csv, osm_on_satellite.png

"Near the holes" filters use golf=hole lines from map.osm in the course folder (OpenStreetMap export),
or a Greenkeeper file (--gkd), or nothing (--buffer 0).

Unity frame: x = metres east of the square's SW corner, z = metres north, y = elevation - y_offset.
"""
import argparse, csv, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
import numpy as np
from scipy import ndimage

CFG = "course.json"
UA = {"User-Agent": "Mozilla/5.0 (coursekit GSPro course build)"}
NOT_VEG = [2, 6, 7, 9, 17, 18]


# ---------------------------------------------------------------- settings
def load_cfg():
    if not os.path.exists(CFG):
        sys.exit("No course.json here. cd into the course folder (or run: coursekit.py new <name> <lat> <lon>).")
    return json.load(open(CFG))


def save_cfg(c):
    json.dump(c, open(CFG, "w"), indent=2)


def utm_epsg(lon):
    zone = int((lon + 180) // 6) + 1
    return 6329 + zone, 26900 + zone, zone          # NAD83(2011) UTM (metres), NAD83 UTM wkid for ArcGIS, zone


def cmd_new(a):
    from pyproj import Transformer
    epsg, wkid, zone = utm_epsg(a.lon)
    e, n = Transformer.from_crs("EPSG:4326", "EPSG:%d" % epsg, always_xy=True).transform(a.lon, a.lat)
    folder = os.path.expanduser(os.path.join(a.root, a.name))
    os.makedirs(folder, exist_ok=True)
    cfg = {"name": a.name, "center_lat": a.lat, "center_lon": a.lon, "size": a.size,
           "epsg": epsg, "arcgis_wkid": wkid, "utm_zone": zone,
           "sw_e": round(e - a.size / 2, 2), "sw_n": round(n - a.size / 2, 2), "y_offset": None}
    if a.sw_e is not None:                           # reuse an existing frame (e.g. Tierra Rejada)
        cfg["sw_e"], cfg["sw_n"] = a.sw_e, a.sw_n
    if a.y_offset is not None:
        cfg["y_offset"] = a.y_offset
    json.dump(cfg, open(os.path.join(folder, CFG), "w"), indent=2)
    print("Created", os.path.join(folder, CFG))
    print(json.dumps(cfg, indent=2))
    print("\nNext:  cd %s  then run lidar, satellite, heightmap. Put an OpenStreetMap export there as map.osm." % folder)


# ---------------------------------------------------------------- downloads
def fetch(url, timeout=600, tries=8):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read(), r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            wait = int(e.headers.get("Retry-After") or 0) or min(120, 10 * 2 ** (k // 2))
            print("    %s, retrying in %ds" % (e.code, wait)); time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print("    network error (%s), retrying in 15s" % e); time.sleep(15)
    raise SystemExit("Gave up on " + url)


def square_bbox_ll(c):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:%d" % c["epsg"], "EPSG:4326", always_xy=True)
    pts = [t.transform(e, n) for e in (c["sw_e"], c["sw_e"] + c["size"]) for n in (c["sw_n"], c["sw_n"] + c["size"])]
    return min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)


def project_of(item):
    u = item.get("downloadURL", "")
    m = re.search(r"/Projects/([^/]+)/", u)
    return m.group(1) if m else item.get("title", "?")


def cmd_lidar(a):
    import laspy
    from pyproj import CRS, Transformer
    c = load_cfg()
    bb = square_bbox_ll(c)
    q = urllib.parse.urlencode({"datasets": "Lidar Point Cloud (LPC)", "bbox": "%f,%f,%f,%f" % bb, "prodFormats": "LAS,LAZ", "max": 500})
    print("Asking USGS which surveys cover the square...")
    items = json.loads(fetch("https://tnmaccess.nationalmap.gov/api/v1/products?" + q, 120)[0]).get("items", [])
    groups = {}
    for it in items:
        groups.setdefault(project_of(it), []).append(it)
    if not groups:
        sys.exit("USGS has no LiDAR point clouds here.")
    def year(p):
        y = re.findall(r"(?:19|20)\d\d", p)
        return max(int(v) for v in y) if y else 0
    ranked = sorted(groups, key=lambda p: (year(p), len(groups[p])), reverse=True)
    print("Surveys found (newest first):")
    for p in ranked:
        print("   %-55s %3d tiles" % (p, len(groups[p])))
    pick = a.survey or ranked[0]
    if pick not in groups:
        sys.exit("No survey named %s above." % pick)
    print("Using", pick, "(override with --survey NAME)")

    os.makedirs("tiles", exist_ok=True)
    utm = CRS.from_epsg(c["epsg"])
    parts = []
    for it in groups[pick]:
        path = os.path.join("tiles", it["downloadURL"].rsplit("/", 1)[-1])
        if not os.path.exists(path):
            print("  downloading", os.path.basename(path), "(%.0f MB)" % ((it.get("sizeInBytes") or 0) / 1e6))
            urls = [it["downloadURL"]]
            if "rockyweb.usgs.gov/vdelivery/Datasets/Staged/" in urls[0]:
                urls.append(urls[0].replace("https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/", "https://prd-tnm.s3.amazonaws.com/StagedProducts/"))
            data = None
            for k in range(12):
                u = urls[k % len(urls)]
                try:
                    with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=600) as r:
                        data = r.read()
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404 and len(urls) > 1 and u != urls[0]:
                        urls = urls[:1]                      # mirror doesn't have this survey; stop trying it
                        print("    mirror doesn't have this tile, using USGS only")
                        continue
                    wait = int(e.headers.get("Retry-After") or 0) or min(300, 30 * (k + 1))
                    print("    %s from %s, waiting %ds" % (e.code, u.split("/")[2], wait)); time.sleep(wait)
                except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                    print("    network error (%s), waiting 30s" % e); time.sleep(30)
            if data is None:
                sys.exit("Could not download %s; rerun later (finished tiles are kept)." % path)
            open(path, "wb").write(data); time.sleep(2)
        las = laspy.read(path)
        try:
            crs = las.header.parse_crs()
        except Exception:
            crs = None
        h = (crs.sub_crs_list[0] if crs is not None and crs.is_compound else crs) or utm
        zs = 1.0
        if crs is not None and crs.is_compound and len(crs.sub_crs_list) > 1:
            u = crs.sub_crs_list[1].axis_info[0].unit_name.lower()
            zs = 1200 / 3937 if "us survey foot" in u else 0.3048 if "foot" in u else 1.0
        xs, ys = np.asarray(las.x), np.asarray(las.y)
        if not h.equals(utm):
            xs, ys = Transformer.from_crs(h, utm, always_xy=True).transform(xs, ys)
        e, n = xs - c["sw_e"], ys - c["sw_n"]
        k = (e >= 0) & (e < c["size"]) & (n >= 0) & (n < c["size"])
        print("    %s: %d points in the square" % (os.path.basename(path), k.sum()))
        parts.append((e[k].astype(np.float32), n[k].astype(np.float32), (np.asarray(las.z)[k] * zs).astype(np.float32),
                      np.asarray(las.classification)[k].astype(np.uint8)))
    x, n_, z, cl = (np.concatenate([p[i] for p in parts]) for i in range(4))
    np.savez_compressed("points.npz", x=x, n=n_, z=z, c=cl)
    g = cl == 2
    print("Saved %d points to points.npz. Ground %.1f to %.1f m." % (len(x), z[g].min(), z[g].max()))
    c["lidar_survey"] = pick
    save_cfg(c)


def cmd_satellite(a):
    from PIL import Image
    import io
    c = load_cfg()
    S, px, tiles = c["size"], a.pixels, 2
    services = ["https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage",
                "https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/exportImage"]
    step, sub = S / tiles, px // tiles
    for svc in services:
        print("Trying", svc.split("/")[2])
        mosaic, ok = Image.new("RGB", (px, px)), True
        for row in range(tiles):
            for col in range(tiles):
                e0, n1 = c["sw_e"] + col * step, c["sw_n"] + S - row * step
                q = urllib.parse.urlencode({"bbox": "%f,%f,%f,%f" % (e0, n1 - step, e0 + step, n1), "bboxSR": c["arcgis_wkid"],
                                            "imageSR": c["arcgis_wkid"], "size": "%d,%d" % (sub, sub), "format": "jpgpng", "f": "image"})
                img = None
                for k in range(5):
                    try:
                        data, ct = fetch(svc + "?" + q, 180, 1)
                        if ct.startswith("image"):
                            img = Image.open(io.BytesIO(data)).convert("RGB"); break
                    except SystemExit:
                        pass
                    time.sleep(5 * (k + 1))
                if img is None:
                    ok = False; break
                mosaic.paste(img.resize((sub, sub)), (col * sub, row * sub))
                print("  piece %d of %d" % (row * tiles + col + 1, tiles * tiles))
            if not ok:
                break
        if ok:
            mosaic.save("satellite.jpg", quality=92)
            print("Saved satellite.jpg (%d px, %.2f m/px). In Unity: Max Size %d, Terrain Size %d." % (px, S / px, px, S))
            return
    sys.exit("Both imagery services failed.")


# ---------------------------------------------------------------- grids
def load_points():
    if not os.path.exists("points.npz"):
        sys.exit("No points.npz yet; run the lidar step first.")
    d = np.load("points.npz")
    return d["x"], d["n"], d["z"], d["c"]


def cell_grid(ix, iy, v, size, how):
    out = np.full((size, size), np.nan, np.float32)
    flat = iy.astype(np.int64) * size + ix
    if how == "mean":
        cnt = np.bincount(flat, minlength=size * size)
        tot = np.bincount(flat, weights=v.astype(np.float64), minlength=size * size)
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).reshape(size, size).astype(np.float32)
    o = np.lexsort((v, flat)); fs, vs = flat[o], v[o]
    ed = np.nonzero(np.diff(fs))[0]
    pick = np.r_[ed, len(fs) - 1] if how == "max" else np.r_[0, ed + 1]
    out.reshape(-1)[fs[pick]] = vs[pick]
    return out


def fill(a):
    bad = np.isnan(a)
    return a[tuple(ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True))] if bad.any() else a


def ground_and_heights(c):
    x, n, z, cl = load_points()
    S = int(c["size"])
    ix, iy = np.clip(x.astype(int), 0, S - 1), np.clip(n.astype(int), 0, S - 1)
    g = cl == 2
    dtm = ndimage.gaussian_filter(fill(cell_grid(ix[g], iy[g], z[g], S, "mean")), 1.0)
    veg = ~np.isin(cl, NOT_VEG)
    under = ndimage.map_coordinates(dtm, [np.clip(n[veg] - 0.5, 0, S - 1), np.clip(x[veg] - 0.5, 0, S - 1)], order=1)
    h = np.nan_to_num(cell_grid(ix[veg], iy[veg], (z[veg] - under).astype(np.float32), S, "max"), nan=0.0)
    h[(h < 0) | (h > 60)] = 0
    return dtm, h


def cmd_heightmap(a):
    c = load_cfg()
    x, n, z, cl = load_points()
    S = int(c["size"])
    g = cl == 2
    ix, iy = np.clip(x[g].astype(int), 0, S - 1), np.clip(n[g].astype(int), 0, S - 1)
    dtm = fill(cell_grid(ix, iy, z[g], S, "mean"))
    res = 2049
    yy, xx = np.mgrid[0:res, 0:res] * (S / (res - 1))
    hm = ndimage.map_coordinates(dtm, [np.clip(yy - 0.5, 0, S - 1), np.clip(xx - 0.5, 0, S - 1)], order=1)
    base = float(np.floor(hm.min()) - 1) if c.get("y_offset") is None else float(c["y_offset"])
    top = float(np.ceil(hm.max() - base) + 1)
    raw = np.clip((hm - base) / top, 0, 1) * 65535
    raw.astype("<u2").tofile("heightmap_2049.raw")      # rows = north, little-endian = Unity "Windows" byte order
    c["y_offset"], c["terrain_height"] = base, top
    save_cfg(c)
    print("Wrote heightmap_2049.raw. Ground %.1f-%.1f m." % (hm.min(), hm.max()))
    print("Unity Terrain: Width %d, Length %d, Height %.0f, Heightmap Resolution 2049; Import Raw: Bit 16, 2049 x 2049, Byte Order Windows."
          % (S, S, top))
    print("Unity y = elevation - %.0f m (saved as y_offset in course.json)." % base)


# ---------------------------------------------------------------- near-the-holes zone
def play_lines(c, gkd=None):
    segs = []
    if gkd and os.path.exists(gkd):
        d = json.load(open(gkd, encoding="utf-8-sig"))
        for hole in d["Holes"]:
            T = {t["TeeType"]: t["Position"] for t in hole["Tees"] if t.get("Position")}
            route = [T[k] for k in ("AimPoint1", "AimPoint2", "GreenCenterPoint") if k in T]
            for k, p in T.items():
                if k not in ("AimPoint1", "AimPoint2", "GreenCenterPoint") and route:
                    q = [p] + route
                    segs += [(q[i]["x"], q[i]["z"], q[i + 1]["x"], q[i + 1]["z"]) for i in range(len(q) - 1)]
        print("Using hole routes from", gkd)
    elif os.path.exists("map.osm"):
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:%d" % c["epsg"], always_xy=True)
        r = ET.parse("map.osm").getroot()
        nodes = {}
        for nd in r.iter("node"):
            e, n = t.transform(float(nd.get("lon")), float(nd.get("lat")))
            nodes[nd.get("id")] = (e - c["sw_e"], n - c["sw_n"])
        holes = 0
        for w in r.findall("way"):
            tg = {q.get("k"): q.get("v") for q in w.findall("tag")}
            if tg.get("golf") == "hole":
                p = [nodes[q.get("ref")] for q in w.findall("nd") if q.get("ref") in nodes]
                segs += [(p[i][0], p[i][1], p[i + 1][0], p[i + 1][1]) for i in range(len(p) - 1)]
                holes += 1
        print("Using %d golf hole lines from map.osm" % holes)
    return segs


def zone_mask(c, buffer, gkd=None):
    S = int(c["size"])
    if buffer <= 0:
        return np.ones((S, S), bool)
    segs = play_lines(c, gkd)
    if not segs:
        print("(no hole lines found, so no zone filter)")
        return np.ones((S, S), bool)
    yy, xx = np.mgrid[0:S, 0:S] + 0.5
    dist = np.full((S, S), 1e9, np.float32)
    for ax_, az, bx, bz in segs:
        dx, dz = bx - ax_, bz - az
        t = np.clip(((xx - ax_) * dx + (yy - az) * dz) / max(dx * dx + dz * dz, 1e-9), 0, 1)
        dist = np.minimum(dist, np.hypot(xx - ax_ - t * dx, yy - az - t * dz))
    return dist <= buffer


def osm_masks(c):
    """Rasterise map.osm: (exclude, course) masks on the 1 m grid (row 0 = south).
    exclude = buildings, parking, roads, residential/commercial land, playing surfaces (greens, tees, bunkers, water).
    course  = the golf course boundary (leisure=golf_course), or None if it isn't mapped."""
    S = int(c["size"])
    if not os.path.exists("map.osm"):
        return None, None
    from pyproj import Transformer
    from PIL import Image, ImageDraw
    t = Transformer.from_crs("EPSG:4326", "EPSG:%d" % c["epsg"], always_xy=True)
    r = ET.parse("map.osm").getroot()
    nodes = {}
    for nd in r.iter("node"):
        e, n = t.transform(float(nd.get("lon")), float(nd.get("lat")))
        nodes[nd.get("id")] = (e - c["sw_e"], S - (n - c["sw_n"]))      # image coords: y down
    ex = Image.new("L", (S, S), 0); dx = ImageDraw.Draw(ex)
    co = Image.new("L", (S, S), 0); dc = ImageDraw.Draw(co)
    road_w = {"motorway": 30, "motorway_link": 14, "trunk": 24, "primary": 20, "secondary": 16, "tertiary": 12,
              "residential": 10, "unclassified": 8, "service": 6}
    has_course = False
    for w in r.findall("way"):
        tg = {q.get("k"): q.get("v") for q in w.findall("tag")}
        p = [nodes[q.get("ref")] for q in w.findall("nd") if q.get("ref") in nodes]
        if len(p) < 2:
            continue
        if tg.get("leisure") == "golf_course" and len(p) > 2:
            dc.polygon(p, fill=255); has_course = True
        elif ("building" in tg or tg.get("amenity") == "parking" or tg.get("landuse") in ("residential", "commercial", "retail", "industrial")
              or tg.get("golf") in ("green", "tee", "bunker", "water_hazard", "lateral_water_hazard")) and len(p) > 2:
            dx.polygon(p, fill=255)
        elif tg.get("highway") in road_w and tg.get("golf") != "cartpath":
            dx.line(p, fill=255, width=road_w[tg["highway"]])
    for rel in r.findall("relation"):                     # course boundary mapped as a multipolygon
        tg = {q.get("k"): q.get("v") for q in rel.findall("tag")}
        if tg.get("leisure") == "golf_course":
            for m in rel.findall("member"):
                if m.get("type") == "way" and m.get("role") in ("outer", ""):
                    w = r.find("./way[@id='%s']" % m.get("ref"))
                    if w is not None:
                        p = [nodes[q.get("ref")] for q in w.findall("nd") if q.get("ref") in nodes]
                        if len(p) > 2:
                            dc.polygon(p, fill=255); has_course = True
    exclude = np.flipud(np.asarray(ex) > 0)
    course = np.flipud(np.asarray(co) > 0) if has_course else None
    print("OSM filter: %.1f ha excluded (buildings, parking, roads, built-up land, greens/tees/bunkers/water)%s"
          % (exclude.sum() / 1e4, ", course boundary found" if has_course else ", no course boundary mapped"))
    return exclude, course


def course_if_valid(c, course):
    """Only trust a mapped course boundary if it actually contains most of the holes."""
    if course is None:
        return None
    segs = play_lines(c, None)
    if not segs:
        return course
    S = int(c["size"])
    pts = [(int(min(max(v, 0), S - 1)), int(min(max(u, 0), S - 1))) for s_ in segs for (u, v) in ((s_[0], s_[1]), (s_[2], s_[3]))]
    share = np.mean([course[r, q] for r, q in pts])
    if share < 0.6:
        print("Ignoring the mapped golf_course boundary (only %.0f%% of the holes are inside it; probably another facility)" % (share * 100))
        return None
    return course


def green_share(c):
    if not os.path.exists("satellite.jpg"):
        return None
    from PIL import Image
    S = int(c["size"])
    sat = np.flipud(np.asarray(Image.open("satellite.jpg").convert("RGB").resize((S, S), Image.BILINEAR), dtype=np.float32))
    return ndimage.uniform_filter(sat[..., 1] / np.maximum(sat.sum(2), 1), 3)


def preview(c, mask_rgba, pts, title, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        return
    S = c["size"]
    fig, ax = plt.subplots(figsize=(18, 18), dpi=110)
    if os.path.exists("satellite.jpg"):
        ax.imshow(Image.open("satellite.jpg").convert("RGB"), origin="upper", extent=(0, S, 0, S))
    if mask_rgba is not None:
        ax.imshow(mask_rgba, origin="lower", extent=(0, S, 0, S))
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=2, c="red")
    ax.set_title(title); ax.set_xlim(0, S); ax.set_ylim(0, S)
    fig.savefig(name, bbox_inches="tight"); print("Wrote", name)


def need_offset(c):
    if c.get("y_offset") is None:
        sys.exit("Run the heightmap step first (it sets y_offset).")
    return c["y_offset"]


def cmd_trees(a):
    c = load_cfg(); yo = need_offset(c); S = int(c["size"])
    dtm, h = ground_and_heights(c)
    hs = ndimage.gaussian_filter(h, 1.0)
    peaks = (hs == ndimage.maximum_filter(hs, size=5)) & (hs >= a.min_height)
    py, px = np.nonzero(peaks)
    order = np.argsort(-hs[py, px]); py, px = py[order], px[order]
    taken, kept = np.zeros((S, S), bool), []
    dirs = [(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    for yy, xx in zip(py, px):
        if taken[yy, xx]:
            continue
        hh, radii = hs[yy, xx], []
        for dx, dy in dirs:
            r = 1
            while r < 15:
                sx, sy = int(round(xx + dx * r)), int(round(yy + dy * r))
                if not (0 <= sx < S and 0 <= sy < S) or hs[sy, sx] < 0.5 * hh:
                    break
                r += 1
            radii.append(r)
        rad = float(np.median(radii)); kept.append((xx, yy, hh, rad))
        rr = max(1, int(round(rad * 0.8)))
        Y, X = np.ogrid[max(0, yy - rr):min(S, yy + rr + 1), max(0, xx - rr):min(S, xx + rr + 1)]
        taken[max(0, yy - rr):min(S, yy + rr + 1), max(0, xx - rr):min(S, xx + rr + 1)] |= (X - xx) ** 2 + (Y - yy) ** 2 <= rr * rr
    zone = zone_mask(c, a.buffer, a.gkd)
    exclude, course = osm_masks(c)
    course = course_if_valid(c, course)
    if exclude is not None:
        zone &= ~ndimage.binary_dilation(exclude, iterations=2)
        if course is not None and a.course_margin >= 0:
            zone &= ndimage.binary_dilation(course, iterations=int(a.course_margin))
    n0 = len(kept)
    kept = [k for k in kept if zone[k[1], k[0]]]
    gs = green_share(c) if a.min_green > 0 else None
    if gs is not None:
        n1 = len(kept)
        kept = [k for k in kept if gs[k[1], k[0]] >= a.min_green]
        print("Satellite check dropped %d grey/tan crowns (roofs, poles, walls)" % (n1 - len(kept)))
    print("Kept %d of %d detected trees after the zone/OSM filters" % (len(kept), n0))
    with open("trees.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "x", "z", "y", "height_m", "crown_radius_m"])
        for i, (xx, yy, hh, rad) in enumerate(kept):
            w.writerow([i, "%.2f" % (xx + .5), "%.2f" % (yy + .5), "%.2f" % (dtm[yy, xx] - yo), "%.1f" % hh, "%.1f" % rad])
    print("Wrote %d trees (>= %.1f m) to trees.csv" % (len(kept), a.min_height))
    preview(c, None, np.array([(k[0] + .5, k[1] + .5) for k in kept]).reshape(-1, 2), "%d trees" % len(kept), "trees_on_satellite.png")


def cmd_brush(a):
    from PIL import Image
    c = load_cfg(); yo = need_offset(c); S = int(c["size"])
    dtm, h = ground_and_heights(c)
    brush = ndimage.binary_opening(ndimage.binary_closing((h >= a.low) & (h <= a.high), iterations=2), iterations=1)
    brush &= zone_mask(c, a.buffer, a.gkd)
    exclude, course = osm_masks(c)
    course = course_if_valid(c, course)
    if exclude is not None:
        brush &= ~ndimage.binary_dilation(exclude, iterations=2)
        if course is not None and a.course_margin >= 0:
            brush &= ndimage.binary_dilation(course, iterations=int(a.course_margin))
    hs = ndimage.gaussian_filter(np.where(brush, h, 0), 0.8)
    win = max(3, int(round(a.spacing)) | 1)
    py, px = np.nonzero(brush & (hs == ndimage.maximum_filter(hs, size=win)) & (hs >= a.low))
    with open("shrubs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "x", "z", "y", "height_m", "crown_radius_m"])
        for i, (yy, xx) in enumerate(zip(py, px)):
            w.writerow([i, "%.2f" % (xx + .5), "%.2f" % (yy + .5), "%.2f" % (dtm[yy, xx] - yo), "%.1f" % h[yy, xx], "%.1f" % (a.spacing / 2)])
    Image.fromarray((np.flipud(brush) * 255).astype(np.uint8)).save("brush_mask.png")
    print("Brush %.1f ha; wrote %d shrubs to shrubs.csv and brush_mask.png" % (brush.sum() / 1e4, len(px)))
    ov = np.zeros((S, S, 4), np.float32); ov[brush] = (1, .45, 0, .45)
    preview(c, ov, np.c_[px + .5, py + .5], "Brush %.1f-%.1f m (orange), %d shrubs" % (a.low, a.high, len(px)), "brush_on_satellite.png")


def cmd_osm(a):
    from pyproj import Transformer
    import collections
    c = load_cfg(); yo = need_offset(c); S = int(c["size"])
    if not os.path.exists("map.osm"):
        sys.exit("No map.osm here. Export the course from openstreetmap.org/export and save it in this folder as map.osm.")
    t = Transformer.from_crs("EPSG:4326", "EPSG:%d" % c["epsg"], always_xy=True)
    r = ET.parse("map.osm").getroot()
    nodes = {}
    for nd in r.iter("node"):
        e, n = t.transform(float(nd.get("lon")), float(nd.get("lat")))
        nodes[nd.get("id")] = (e - c["sw_e"], n - c["sw_n"])
    ways = []
    for w in r.findall("way"):
        tg = {q.get("k"): q.get("v") for q in w.findall("tag")}
        ways.append((tg, [nodes[q.get("ref")] for q in w.findall("nd") if q.get("ref") in nodes]))
    golf = collections.Counter(tg["golf"] for tg, _ in ways if "golf" in tg)
    print("Golf features in map.osm:", dict(golf))
    print("Buildings:", sum(1 for tg, _ in ways if "building" in tg),
          " Parking areas:", sum(1 for tg, _ in ways if tg.get("amenity") == "parking"),
          " Fences/walls:", sum(1 for tg, _ in ways if tg.get("barrier") in ("fence", "wall", "retaining_wall")))
    def length(p): return sum(np.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1]) for i in range(len(p) - 1))
    carts = [p for tg, p in ways if tg.get("golf") == "cartpath" and len(p) > 1]
    if not carts and a.paths_fallback:
        carts = [p for tg, p in ways if tg.get("highway") in ("path", "footway", "track") and len(p) > 1]
        print("No golf=cartpath ways; using %d paths/footways/tracks instead (check them!)" % len(carts))
    x, n, z, cl = load_points(); g = cl == 2
    dtm = fill(cell_grid(np.clip(x[g].astype(int), 0, S - 1), np.clip(n[g].astype(int), 0, S - 1), z[g], S, "mean"))
    with open("cartpaths.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["segment", "x", "z", "y"])
        for i, p in enumerate(carts):
            for ex, nz in p:
                yy = dtm[int(min(max(nz, 0), S - 1)), int(min(max(ex, 0), S - 1))] - yo
                w.writerow([i, "%.2f" % ex, "%.2f" % nz, "%.2f" % yy])
    print("Cart paths: %d, %.0f m in total -> cartpaths.csv" % (len(carts), sum(length(p) for p in carts)))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        fig, ax = plt.subplots(figsize=(18, 18), dpi=110)
        if os.path.exists("satellite.jpg"):
            ax.imshow(Image.open("satellite.jpg").convert("RGB"), origin="upper", extent=(0, S, 0, S))
        cols = {"green": "#2f9e44", "tee": "#8ce99a", "bunker": "#ffe066", "fairway": "#69db7c", "water_hazard": "#4dabf7", "rough": "#d3f9d8"}
        for tg, p in ways:
            if len(p) < 2:
                continue
            q = np.array(p)
            if tg.get("golf") in cols:
                ax.fill(q[:, 0], q[:, 1], color=cols[tg["golf"]], alpha=0.5)
            elif tg.get("golf") == "hole":
                ax.plot(q[:, 0], q[:, 1], "w--", lw=1); ax.text(q[0, 0], q[0, 1], tg.get("ref", ""), color="yellow", fontsize=12, weight="bold")
            elif "building" in tg:
                ax.fill(q[:, 0], q[:, 1], color="#868e96", alpha=0.8)
        for p in carts:
            q = np.array(p); ax.plot(q[:, 0], q[:, 1], color="red", lw=2)
        ax.set_xlim(0, S); ax.set_ylim(0, S)
        ax.set_title("OpenStreetMap: cart paths (red), greens/tees/fairways/bunkers/water, holes (dashed), buildings (grey)")
        fig.savefig("osm_on_satellite.png", bbox_inches="tight"); print("Wrote osm_on_satellite.png")
    except ImportError:
        pass


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("new"); p.add_argument("name"); p.add_argument("lat", type=float); p.add_argument("lon", type=float)
    p.add_argument("--size", type=float, default=2000); p.add_argument("--root", default="~/courses")
    p.add_argument("--sw-e", type=float); p.add_argument("--sw-n", type=float); p.add_argument("--y-offset", type=float)
    p = sp.add_parser("lidar"); p.add_argument("--survey")
    p = sp.add_parser("satellite"); p.add_argument("--pixels", type=int, default=4096)
    sp.add_parser("heightmap")
    p = sp.add_parser("trees"); p.add_argument("--min-height", type=float, default=3.0); p.add_argument("--buffer", type=float, default=150); p.add_argument("--gkd")
    p.add_argument("--min-green", type=float, default=0.355, help="satellite green share a crown needs (0 = off)")
    p.add_argument("--course-margin", type=float, default=60, help="metres beyond the mapped course boundary to keep (-1 = ignore boundary)")
    p = sp.add_parser("brush"); p.add_argument("--low", type=float, default=0.5); p.add_argument("--high", type=float, default=3.5)
    p.add_argument("--spacing", type=float, default=3.0); p.add_argument("--buffer", type=float, default=200); p.add_argument("--gkd")
    p.add_argument("--course-margin", type=float, default=60, help="metres beyond the mapped course boundary to keep (-1 = ignore boundary)")
    p = sp.add_parser("osm"); p.add_argument("--paths-fallback", action="store_true", help="use footpaths if no golf=cartpath")
    a = ap.parse_args()
    {"osm": cmd_osm, "new": cmd_new, "lidar": cmd_lidar, "satellite": cmd_satellite, "heightmap": cmd_heightmap,
     "trees": cmd_trees, "brush": cmd_brush}[a.cmd](a)


if __name__ == "__main__":
    main()
