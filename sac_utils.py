from dataclasses import dataclass, asdict, is_dataclass
import numpy as np, torch, os

@dataclass
class ParsedObs:
    img: np.ndarray; vec: np.ndarray; limits: np.ndarray
    reached: bool=False; collided: bool=False; no_path: bool=False

def _to_dict(obs):
    if is_dataclass(obs): return asdict(obs)
    if isinstance(obs, dict): return obs
    keys = ["state","goal","limits","esdf_local"]
    return {k: getattr(obs, k, None) for k in keys}

def normalize_esdf_patch(esdf_local: np.ndarray, safety_radius: float = 0.5) -> np.ndarray:
    if esdf_local is None: return np.zeros((32,32), np.float32)
    esdf = esdf_local.astype(np.float32)
    scale = max(1e-6, 3.0 * float(safety_radius))
    return np.clip(esdf / scale, 0.0, 1.0)

def parse_obs(obs) -> ParsedObs:
    d = _to_dict(obs)
    state  = np.asarray(d.get("state"),  dtype=np.float32).reshape(-1)
    goal   = np.asarray(d.get("goal"),   dtype=np.float32).reshape(-1)
    limits = np.asarray(d.get("limits"), dtype=np.float32).reshape(-1)
    esdf_local = d.get("esdf_local")
    x,y,theta,*_ = list(state) + [0,0,0,0]
    gx,gy = (goal.tolist() + [0,0])[:2]
    g_rel = np.array([gx-x, gy-y], np.float32)
    bearing = np.arctan2(g_rel[1], g_rel[0]) - theta
    bearing = (bearing + np.pi) % (2*np.pi) - np.pi
    vec = np.concatenate([state, goal, g_rel, np.array([bearing], np.float32)], axis=0).astype(np.float32)
    img = np.expand_dims(normalize_esdf_patch(esdf_local, safety_radius=0.5), axis=0)
    return ParsedObs(img=img, vec=vec, limits=limits,
                     reached=bool(d.get("reached", False)),
                     collided=bool(d.get("collided", False)),
                     no_path=bool(d.get("no_path", False)))

def to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, device=device, dtype=dtype)

def save_checkpoint(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import torch; torch.save(payload, path)

def load_checkpoint(path: str):
    import os, torch
    if os.path.isfile(path): return torch.load(path, map_location="cpu",weights_only=False)
    return None

def evaluate(env, actor, parse_obs_fn=parse_obs, episodes: int = 5, device: str = "cpu"):
    import torch
    actor.eval()
    success = collisions = 0; lengths = []
    for _ in range(episodes):
        obs, _ = env.reset(); steps = 0; done = False
        while not done:
            p = parse_obs_fn(obs)
            with torch.no_grad():
                img = to_tensor(p.img[None], device)
                vec = to_tensor(p.vec[None], device)
                limits = to_tensor(p.limits[None], device)
                mu, _ = actor.forward(img, vec)
                a = torch.tanh(mu) * limits
            obs, _, term, trunc, inf = env.step(a.cpu().numpy()[0])
            done = bool(term or trunc); steps += 1
        lengths.append(steps)
        success += int(bool(inf.get("reached", False)))
        collisions += int(bool(inf.get("collided", False)))
    actor.train()
    return {"success_rate": success/max(1,episodes),
            "avg_len": sum(lengths)/max(1,len(lengths)),
            "collisions": collisions/max(1,episodes)}