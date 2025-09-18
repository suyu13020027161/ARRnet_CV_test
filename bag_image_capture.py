import pyrealsense2 as rs
import numpy as np
import cv2
import os
from bisect import bisect_left

#Change it to your own .bag file path!!!
bag_file = "20250806_114323.bag"
output_dir = "dataset"
num_samples = 50
os.makedirs(output_dir, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
config.enable_stream(rs.stream.color) 
profile = pipeline.start(config)
playback = profile.get_device().as_playback()
playback.set_real_time(False) 
frames_ts = []    
frames_img = []   
try:
    while True:
        try:
            fs = pipeline.wait_for_frames(timeout_ms=2000)
        except RuntimeError:
            break
        color = fs.get_color_frame()
        if not color:
            continue
        ts_ms = float(color.get_timestamp())
        img = np.asanyarray(color.get_data())

        frames_ts.append(ts_ms)
        frames_img.append(img.copy())
finally:
    pipeline.stop()

n = len(frames_ts)
if n == 0:
    raise RuntimeError("No RGB image!")
if n == 1:
    out = os.path.join(output_dir, "frame_000.jpg")
    cv2.imwrite(out, frames_img[0])
    print(f"only 1 frame!")
    raise SystemExit(0)
order = np.argsort(frames_ts)
frames_ts = np.asarray(frames_ts, dtype=np.float64)[order]
frames_img = [frames_img[i] for i in order]

t0, t1 = frames_ts[0], frames_ts[-1]
targets = np.linspace(t0, t1, num_samples)
indices = []
last_idx = -1
for t in targets:
    pos = bisect_left(frames_ts, t)
    if pos == 0:
        idx = 0
    elif pos >= n:
        idx = n - 1
    else:
        left, right = pos - 1, pos
        if abs(frames_ts[right] - t) < abs(frames_ts[left] - t):
            idx = right
        else:
            idx = left
    if idx == last_idx:
        if idx < n - 1:
            idx += 1
        elif last_idx > 0:
            idx = last_idx - 1
    if indices and idx == indices[-1]:
        forward = idx + 1
        while forward < n and (len(indices) > 0 and forward == indices[-1]):
            forward += 1
        if forward < n:
            idx = forward
        else:
            backward = idx - 1
            while backward >= 0 and (len(indices) > 0 and backward == indices[-1]):
                backward -= 1
            if backward >= 0:
                idx = backward
    indices.append(idx)
    last_idx = idx
uniq = []
seen = set()
for i in indices:
    if i not in seen:
        uniq.append(i)
        seen.add(i)

if len(uniq) < num_samples:
    lin_idx = np.linspace(0, n - 1, num_samples)
    fill = np.unique(np.round(lin_idx).astype(int))
    merged = sorted(set(uniq).union(set(fill)))
    take = np.linspace(0, len(merged) - 1, num_samples)
    indices = [merged[int(round(x))] for x in take]
else:
    indices = uniq[:num_samples]
indices = sorted(set(indices))
if len(indices) < num_samples:
    indices = np.unique(np.round(np.linspace(0, n - 1, num_samples)).astype(int)).tolist()
for k, idx in enumerate(indices[:num_samples]):
    img = frames_img[idx]
    out_path = os.path.join(output_dir, f"frame_{k:03d}.jpg")
    cv2.imwrite(out_path, img)


