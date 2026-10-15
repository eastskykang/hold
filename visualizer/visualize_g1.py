#!/usr/bin/env python3
"""Play HOLD G1 trajectories (data/g1/**/*.h5) in the MuJoCo viewer.

Usage:
    uv run --project visualizer visualizer/visualize_g1.py data/g1/locomotion/walk
    uv run --project visualizer visualizer/visualize_g1.py data/g1/locomotion/sprint/s0_sprint_001.h5

Keys: SPACE pause/resume, N next motion, P previous motion, R restart,
      Z slower, X faster.
"""

import argparse
import re
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

import h5py
import mujoco
import mujoco.viewer
import numpy as np

# Unitree G1 (29 DoF) model from MuJoCo Menagerie, downloaded once and cached.
MENAGERIE_COMMIT = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
MENAGERIE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    f"{MENAGERIE_COMMIT}/unitree_g1"
)
DEFAULT_CACHE = Path.home() / ".cache" / "hold" / f"unitree_g1_{MENAGERIE_COMMIT[:8]}"

KEY_SPACE, KEY_N, KEY_P, KEY_R, KEY_Z, KEY_X = 32, ord("N"), ord("P"), ord("R"), ord("Z"), ord("X")


def fetch_model(cache_dir: Path) -> Path:
    """Returns the path to scene.xml, downloading the G1 model if needed."""
    scene = cache_dir / "scene.xml"
    if scene.exists():
        return scene

    print(f"Downloading the Unitree G1 model to {cache_dir} ...")
    tmp = Path(tempfile.mkdtemp(prefix="g1_model_"))

    def download(name: str) -> None:
        dst = tmp / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{MENAGERIE_URL}/{name}", dst)

    try:
        for name in ("g1.xml", "scene.xml", "LICENSE"):
            download(name)
        for mesh in sorted(set(re.findall(r'<mesh\b[^>]*\bfile="([^"]+)"', (tmp / "g1.xml").read_text()))):
            download(f"assets/{mesh}")
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.move(str(tmp), cache_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return scene


class Motion:
    """A HOLD G1 trajectory converted to MuJoCo qpos (free joint on the pelvis)."""

    def __init__(self, path: Path, model: mujoco.MjModel):
        self.path = path
        with h5py.File(path, "r") as f:
            traj = f["robot_data/trajectory_data"]
            self.time = traj["time"][:]
            joint_pos = traj["joints/pos"][:]
            joint_names = [n.decode() for n in traj["joints/labels"][:]]
            body_names = [n.decode() for n in traj["bodies/labels"][:]]
            base = body_names.index("base")
            base_pos = traj["bodies/pos_w"][:, base]
            base_quat = traj["bodies/quat_w"][:, base]  # [w, x, y, z]

        # Joint order differs between files, so map by name.
        qadr = [model.joint(n).qposadr[0] for n in joint_names]
        self.qpos = self._to_qpos(model, qadr, joint_pos, base_pos, base_quat)

    @staticmethod
    def _to_qpos(model, qadr, joint_pos, base_pos, base_quat) -> np.ndarray:
        # The "base" frame in the data coincides with the G1 torso_link, whereas the
        # MJCF root is the pelvis. Recover the pelvis pose from the torso pose using
        # forward kinematics of the waist joints.
        data = mujoco.MjData(model)
        torso = model.body("torso_link").id
        qpos = np.zeros((len(joint_pos), model.nq))
        torso_quat_inv = np.zeros(4)
        pelvis_quat = np.zeros(4)
        offset = np.zeros(3)
        for i in range(len(joint_pos)):
            data.qpos[:] = 0.0
            data.qpos[3] = 1.0
            data.qpos[qadr] = joint_pos[i]
            mujoco.mj_kinematics(model, data)
            mujoco.mju_negQuat(torso_quat_inv, data.xquat[torso])
            mujoco.mju_mulQuat(pelvis_quat, base_quat[i], torso_quat_inv)
            mujoco.mju_rotVecQuat(offset, data.xpos[torso], pelvis_quat)
            qpos[i, qadr] = joint_pos[i]
            qpos[i, :3] = base_pos[i] - offset
            qpos[i, 3:7] = pelvis_quat
        return qpos

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    def sample(self, t: float) -> np.ndarray:
        """Returns qpos at time t, interpolated between the two surrounding frames."""
        i = min(int(np.searchsorted(self.time, t, side="right")) - 1, len(self.time) - 2)
        i = max(i, 0)
        a = float(np.clip((t - self.time[i]) / (self.time[i + 1] - self.time[i]), 0.0, 1.0))
        q0, q1 = self.qpos[i], self.qpos[i + 1]
        qpos = (1.0 - a) * q0 + a * q1
        if np.dot(q0[3:7], q1[3:7]) < 0.0:  # take the short way around for the root quaternion
            qpos[3:7] = (1.0 - a) * q0[3:7] - a * q1[3:7]
        qpos[3:7] /= np.linalg.norm(qpos[3:7])
        return qpos


def collect_files(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files += sorted(p.rglob("*.h5"))
        elif p.suffix == ".h5":
            files.append(p)
        else:
            raise SystemExit(f"Not an .h5 file or directory: {p}")
    if not files:
        raise SystemExit("No .h5 files found.")
    return files


class Player:
    def __init__(self, model, files, speed):
        self.model = model
        self.data = mujoco.MjData(model)
        self.files = files
        self.speed = speed
        self.index = 0
        self.paused = False
        self.motion = None
        self.clock = 0.0  # motion time in seconds
        self._last = time.perf_counter()
        self._load()

    def _load(self) -> None:
        path = self.files[self.index]
        print(f"[{self.index + 1}/{len(self.files)}] {path}")
        motion = Motion(path, self.model)
        self.motion = motion
        self.clock = 0.0
        self._last = time.perf_counter()  # don't count the loading time as playback time

    def key_callback(self, key: int) -> None:
        if key == KEY_SPACE:
            self.paused = not self.paused
        elif key == KEY_R:
            self.clock = 0.0
        elif key in (KEY_N, KEY_P):
            step = 1 if key == KEY_N else -1
            self.index = (self.index + step) % len(self.files)
            self._load()
        elif key == KEY_Z:
            self.speed = max(self.speed / 1.25, 0.05)
        elif key == KEY_X:
            self.speed = min(self.speed * 1.25, 8.0)

    def step(self) -> None:
        now = time.perf_counter()
        if not self.paused:
            # Motion time follows the timestamps stored in the H5 file (real time at 1x).
            # Cap dt so a stall (e.g. window startup) doesn't skip ahead in the motion.
            self.clock += min(now - self._last, 0.1) * self.speed
            self.clock %= self.motion.duration
        self._last = now
        self.data.qpos[:] = self.motion.sample(self.clock)
        mujoco.mj_kinematics(self.model, self.data)

    def overlay(self) -> tuple:
        m = self.motion
        status = "paused" if self.paused else "playing"
        left = "File\nTime\nSpeed\nStatus"
        right = (
            f"[{self.index + 1}/{len(self.files)}] {m.path.name}\n"
            f"{self.clock:5.2f} / {m.duration:.2f} s\n{self.speed:.2f}x\n{status}"
        )
        return (None, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="H5 files and/or directories to search for .h5 files")
    parser.add_argument("--speed", type=float, default=1.0, help="initial playback speed (default: 1.0)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_CACHE,
                        help="directory with the G1 MJCF (scene.xml); downloaded here if missing")
    args = parser.parse_args()

    files = collect_files(args.paths)
    model = mujoco.MjModel.from_xml_path(str(fetch_model(args.model_dir)))
    player = Player(model, files, args.speed)

    with mujoco.viewer.launch_passive(
        model, player.data, key_callback=player.key_callback, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.0, -15.0, 140.0
        while viewer.is_running():
            with viewer.lock():
                player.step()
                viewer.cam.lookat[:] = player.data.qpos[:3]  # follow the pelvis; rotate/zoom stay free
            viewer.set_texts(player.overlay())
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
