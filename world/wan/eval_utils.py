"""Small, dependency-light helpers shared by navigation evaluators."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import quote


def episode_key(episode, include_scene=False):
    key = str(episode.episode_id)
    if include_scene:
        scene = os.path.basename(episode.scene_id).split('.')[0]
        key = f"{scene}_{key}"
    return key


def valid_result(path, key):
    try:
        with open(path) as f:
            result = json.load(f)
        if str(result['id']) != key:
            return False
        for name in ('success', 'spl', 'distance_to_goal'):
            value = result[name]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return False
        return result['success'] in (0, 1) and 0 <= result['spl'] <= 1 and result['distance_to_goal'] >= 0
    except (OSError, ValueError, KeyError, TypeError):
        return False


def pending_dataset(dataset, result_path, include_scene=False):
    """Filter before constructing Habitat Env, without mutating the input dataset."""
    pending = copy.copy(dataset)
    pending.episodes = [ep for ep in dataset.episodes if not valid_result(
        Path(result_path) / 'log' / f'stats_{episode_key(ep, include_scene)}.json',
        episode_key(ep, include_scene),
    )]
    print(f'[Resume] completed={len(dataset.episodes) - len(pending.episodes)} '
          f'pending={len(pending.episodes)} total={len(dataset.episodes)}')
    return pending


def write_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(result, f, indent=4)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_future_video(video, result_path, episode_id, step):
    """Save the [1,T,H,W,3] float [0,1] prediction only when explicitly called."""
    import numpy as np
    import imageio.v2 as imageio

    frames = np.asarray(video)
    if frames.ndim != 5 or frames.shape[0] != 1 or frames.shape[-1] != 3 or frames.shape[1] == 0:
        raise ValueError(f'Unexpected future-frame shape: {frames.shape}')
    if not np.isfinite(frames).all():
        raise ValueError('Future frames contain nonfinite values')
    frames = np.clip(frames[0] * 255, 0, 255).astype(np.uint8)
    directory = Path(result_path) / 'future_videos' / quote(str(episode_id), safe='')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'step_{step:04d}.mp4'
    fd, temporary = tempfile.mkstemp(prefix=target.stem + '.', suffix='.mp4', dir=directory)
    os.close(fd)
    try:
        imageio.mimwrite(temporary, frames, fps=5, codec='libx264', macro_block_size=1)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(target)
