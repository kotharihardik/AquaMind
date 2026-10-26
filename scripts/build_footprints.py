"""Hydraulic footprints of a leak (ADR-010): for a leak of size q at every junction j of the NOMINAL model, how much faster does
water move in every pipe, and where does the flow direction reverse?  (Sediment on pipe walls is mobilised when the shear / velocity
exceeds what the pipe has been used to, or when the flow reverses: the brown-water mechanism.)

Output data/ltown/footprints.npz:
  sizes_lps [Q]; pipes [P]; ratio [J, Q, P] float16 = max_t |v_leak| / max_t |v_nominal| (day 2 of a 48 h run);
  reverse [J, Q, P] bool = the night flow direction flips; base_vmax [P] m/s; pipe_ends [P, 2] junction indices (-1 if not a junction);
  pipe_mid [P, 2] metres.
usage: python -I scripts/build_footprints.py [--workers 3] [--sizes 2,4,8]
"""
import argparse, copy, os, shutil, tempfile, time, warnings
from multiprocessing import Pool
import numpy as np, wntr
warnings.filterwarnings("ignore")
INP = "data/raw/ltown/L-TOWN.inp"
_G = {}

def _run(wn):
    d = tempfile.mkdtemp(prefix="agx_")
    try:
        return wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(d, "s"))
    finally:
        shutil.rmtree(d, ignore_errors=True)

def _init():
    wn = wntr.network.WaterNetworkModel(INP)
    wn.options.time.duration = 48 * 3600; wn.options.time.report_timestep = 900; wn.options.time.hydraulic_timestep = 900
    base = _run(wn)
    d2 = lambda df: df[df.index >= 24 * 3600]
    V, F = d2(base.link["velocity"]).abs(), d2(base.link["flowrate"])
    pipes = [p for p in V.columns if wn.get_link(p).link_type == "Pipe"]
    night = (F.index % 86400 >= 2 * 3600) & (F.index % 86400 < 4 * 3600)
    _G.update(wn=wn, pipes=pipes, vmax=V[pipes].max().values, f0=F[pipes][night].mean().values, night=night)

def _one(args):
    j, q = args
    wn = copy.deepcopy(_G["wn"]); wn.get_node(j).add_demand(q / 1000, None, "leak")
    r = _run(wn); pipes = _G["pipes"]
    V = r.link["velocity"].abs(); F = r.link["flowrate"]
    V = V[V.index >= 24 * 3600][pipes]; F = F[F.index >= 24 * 3600][pipes]
    night = (F.index % 86400 >= 2 * 3600) & (F.index % 86400 < 4 * 3600)
    ratio = V.max().values / np.maximum(_G["vmax"], 1e-4)
    rev = (np.abs(_G["f0"]) > 1e-5) & (F[night].mean().values * _G["f0"] < 0)
    return j, q, ratio.astype(np.float16), rev

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--sizes", default="2,4,8")
    a = ap.parse_args(); sizes = [float(x) for x in a.sizes.split(",")]
    _init(); wn = _G["wn"]; pipes = _G["pipes"]; junc = wn.junction_name_list
    jidx = {n: i for i, n in enumerate(junc)}
    ratio = np.zeros((len(junc), len(sizes), len(pipes)), np.float16); reverse = np.zeros((len(junc), len(sizes), len(pipes)), bool)
    jobs = [(j, q) for j in junc for q in sizes]; t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for k, (j, q, r, rv) in enumerate(pool.imap_unordered(_one, jobs, chunksize=8), 1):
            ratio[jidx[j], sizes.index(q)] = r; reverse[jidx[j], sizes.index(q)] = rv
            if k % 200 == 0: print(f"{k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    xy = {n: np.array(wn.get_node(n).coordinates) for n in wn.node_name_list}
    ends = np.array([[jidx.get(wn.get_link(p).start_node_name, -1), jidx.get(wn.get_link(p).end_node_name, -1)] for p in pipes])
    mid = np.array([(xy[wn.get_link(p).start_node_name] + xy[wn.get_link(p).end_node_name]) / 2 for p in pipes])
    np.savez_compressed("data/ltown/footprints.npz", sizes_lps=np.array(sizes), pipes=np.array(pipes), ratio=ratio, reverse=reverse,
                        base_vmax=_G["vmax"], pipe_ends=ends, pipe_mid=mid, junctions=np.array(junc))
    print("done", time.time() - t0)
