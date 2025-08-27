#支持 --config/--session 与 --use-env-v2

import argparse, json, numpy as np, torch
import torch.optim as optim
from rl_sac.sac_models import Actor, Critic
from rl_sac.sac_buffer import ReplayBuffer
from rl_sac.sac_utils import parse_obs, to_tensor, save_checkpoint, load_checkpoint, evaluate

def load_envs(use_env_v2: bool, cfg_path: str, sess_path: str | None):
    from env.config import EnvConfig
    cfg = EnvConfig.from_json(cfg_path)
    if use_env_v2:
        from env.env_core_v2 import EnvV2
        from env.session import SessionConfig, AutoCriteria
        if sess_path:
            js = json.load(open(sess_path, "r", encoding="utf-8"))
            auto = js.get("auto", {})
            sc = SessionConfig(
                tasks_mode=js.get("tasks_mode","fixed"),
                tasks_unit=js.get("tasks_unit","meters"),
                tasks=js.get("tasks"),
                initial_heading=js.get("initial_heading","zero"),
                advance_mode=js.get("advance_mode","manual"),
                auto=AutoCriteria(window=int(auto.get("window",100)),
                                  success_rate=float(auto.get("success_rate",0.9)),
                                  min_episodes=int(auto.get("min_episodes",50))),
                sample_k=int(js.get("sample_k",16)),
                min_start_goal_m=float(js.get("min_start_goal_m",15.0)),
            )
        else:
            sc = SessionConfig(tasks_mode="sample_once", sample_k=16, min_start_goal_m=15.0, advance_mode="auto")
        return EnvV2(cfg, sc)
    else:
        from env.env_core import Env
        return Env(cfg)

def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="env_config.json")
    p.add_argument("--session", type=str, default=None)
    p.add_argument("--use-env-v2", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=200000)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--ckpt", type=str, default="runs/sac_ckpt.pt")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(args)

    np.random.seed(a.seed); torch.manual_seed(a.seed)
    env = load_envs(a.use_env_v2, a.config, a.session)
    obs, _ = env.reset()

    p0 = parse_obs(obs)
    H,W = p0.img.shape[-2:]; vec_dim = p0.vec.size; act_dim = p0.limits.size

    actor = Actor(vec_dim, act_dim, img_ch=1).to(a.device)
    critic1 = Critic(vec_dim, act_dim, img_ch=1).to(a.device)
    critic2 = Critic(vec_dim, act_dim, img_ch=1).to(a.device)
    target1 = Critic(vec_dim, act_dim, img_ch=1).to(a.device)
    target2 = Critic(vec_dim, act_dim, img_ch=1).to(a.device)
    target1.load_state_dict(critic1.state_dict()); target2.load_state_dict(critic2.state_dict())

    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_c1 = optim.Adam(critic1.parameters(), lr=3e-4)
    opt_c2 = optim.Adam(critic2.parameters(), lr=3e-4)
    log_alpha = torch.tensor(np.log(0.2), device=a.device, requires_grad=True)
    opt_alpha = optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -float(act_dim)

    buf = ReplayBuffer(
        1_000_000, img_shape=(H, W), vec_dim=vec_dim, act_dim=act_dim,
        disk_dir="runs/replay_mm",   # 存放 s_img/s2_img 的 memmap 文件
        img_dtype="uint8",           # SSD 占用最省（两份图像合计≈18.4GB @96x96, 1e6）
        # mmap_mode="w+",            # 可选："w+" 每次新建；"r+" 复用已有文件
    )

    if a.resume:
        payload = load_checkpoint(a.ckpt)
        if payload is not None:
            actor.load_state_dict(payload["actor"])
            critic1.load_state_dict(payload["critic1"])
            critic2.load_state_dict(payload["critic2"])
            target1.load_state_dict(payload["target1"])
            target2.load_state_dict(payload["target2"])
            log_alpha.data = torch.tensor(payload["log_alpha"], dtype=torch.float64, device=log_alpha.device)
            np.random.set_state(payload["np_rng_state"])
            torch.random.set_rng_state(payload["torch_rng_state"])

    total = 0; obs = obs
    q_s, pi_s, alpha_s = None, None, None
    while total < a.max_steps:
        # 探索
        if total < 3000:
            act = (np.random.rand(act_dim).astype(np.float32) * 2 - 1) * p0.limits
        else:
            parsed = parse_obs(obs)
            with torch.no_grad():
                img = to_tensor(parsed.img[None], a.device)
                vec = to_tensor(parsed.vec[None], a.device)
                limits = to_tensor(parsed.limits[None], a.device)
                act, _ = actor.sample(img, vec, limits)
            act = act.cpu().numpy()[0]

        obs2, r, term, trunc, info = env.step(act)
        done = bool(term or trunc)

        p1 = parse_obs(obs); p2 = parse_obs(obs2)
        buf.add(p1.img, p1.vec, act.astype(np.float32), np.array([r], np.float32),
                p2.img, p2.vec, np.array([float(done)], np.float32), p1.limits, p2.limits)
        obs = obs2; total += 1

        if done:
            if hasattr(env, "report_episode"):
                try: env.report_episode(bool(info.get("reached", False)))
                except Exception: pass
            obs, _ = env.reset()

        # 更新
        if total >= 1000 and buf.size() >= 256:
            B = 256; batch = buf.sample(B)
            s_img  = to_tensor(batch.s_img, a.device);  s_vec  = to_tensor(batch.s_vec, a.device)
            a_t    = to_tensor(batch.a, a.device);      r_t    = to_tensor(batch.r, a.device)
            s2_img = to_tensor(batch.s2_img, a.device); s2_vec = to_tensor(batch.s2_vec, a.device)
            d_t    = to_tensor(batch.done, a.device)
            lim1   = to_tensor(batch.limits, a.device)
            lim2   = to_tensor(batch.limits2, a.device)

            with torch.no_grad():
                a2, logp2 = actor.sample(s2_img, s2_vec, lim2)
                q1_t = target1(s2_img, s2_vec, a2)
                q2_t = target2(s2_img, s2_vec, a2)
                y = r_t + (1.0 - d_t) * 0.99 * (torch.min(q1_t, q2_t) - log_alpha.exp()*logp2)

            q1 = critic1(s_img, s_vec, a_t)
            q2 = critic2(s_img, s_vec, a_t)
            loss_q = (q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()
            opt_c1.zero_grad(set_to_none=True); opt_c2.zero_grad(set_to_none=True)
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(critic1.parameters(), 10.0)
            torch.nn.utils.clip_grad_norm_(critic2.parameters(), 10.0)
            opt_c1.step(); opt_c2.step()

            a_s, logp_s = actor.sample(s_img, s_vec, lim1)
            q_min = torch.min(critic1(s_img, s_vec, a_s), critic2(s_img, s_vec, a_s))
            loss_pi = (log_alpha.exp()*logp_s - q_min).mean()
            opt_actor.zero_grad(set_to_none=True); loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0); opt_actor.step()

            loss_alpha = -(log_alpha * (logp_s.detach() + target_entropy)).mean()
            opt_alpha.zero_grad(set_to_none=True); loss_alpha.backward(); opt_alpha.step()

            # 软更新
            tau = 5e-3
            for tgt, src in [(target1, critic1), (target2, critic2)]:
                for tp, sp in zip(tgt.parameters(), src.parameters()):
                    tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

            # 简单 EMA 日志
            qm = float(q_min.mean().item()); pm = float(loss_pi.item()); am = float(log_alpha.exp().item())
            q_s = 0.95*(q_s if q_s is not None else qm) + 0.05*qm
            pi_s = 0.95*(pi_s if pi_s is not None else pm) + 0.05*pm
            alpha_s = 0.95*(alpha_s if alpha_s is not None else am) + 0.05*am

        if total % a.eval_every == 0:
            stats = evaluate(env, actor, episodes=5, device=a.device)
            print(f"[{total:>7}] eval: succ={stats['success_rate']:.2f} "
                  f"avg_len={stats['avg_len']:.1f} coll={stats['collisions']:.2f} | "
                  f"Q~{q_s:.2f} PiLoss~{pi_s:.3f} alpha~{alpha_s:.3f}")
            save_checkpoint(a.ckpt, {
                "actor": actor.state_dict(), "critic1": critic1.state_dict(), "critic2": critic2.state_dict(),
                "target1": target1.state_dict(), "target2": target2.state_dict(),
                "log_alpha": log_alpha.detach().cpu().numpy(),
                "np_rng_state": np.random.get_state(),
                "torch_rng_state": torch.random.get_rng_state(),
            })
            obs, _ = env.reset()

    stats = evaluate(env, actor, episodes=10, device=a.device)
    print(f"[final] succ={stats['success_rate']:.2f}, avg_len={stats['avg_len']:.1f}, coll={stats['collisions']:.2f}")
    print(f"checkpoint saved to: {a.ckpt}")

if __name__ == "__main__":
    main()
