## 속도 (6채널, JPEG 디코딩 제외, 중앙값 ms)

측정 환경: {'cpu_count': 12, 'loadavg_1_5_15': [20.28, 18.36, 13.32]}, cv2 threads=12

### 워핑 — 800x800 (5 cm/px)

| method | median_ms | p10_ms | p90_ms |
|---|---|---|---|
| per_frame_coords | 28.5 | 23.9 | 36.6 |
| warpPerspective | 3.4 | 3.2 | 4.4 |
| remap_float | 6.4 | 5.1 | 8.9 |
| remap_fixed | 6.6 | 5.9 | 12.2 |
| remap_fixed_half | 6.6 | 5.8 | 7.6 |

### 워핑 — 400x400 (10 cm/px)

| method | median_ms | p10_ms | p90_ms |
|---|---|---|---|
| per_frame_coords | 6.9 | 6.4 | 7.6 |
| warpPerspective | 2.4 | 2.2 | 3.1 |
| remap_float | 3.1 | 2.9 | 4.4 |
| remap_fixed | 2.9 | 2.8 | 3.5 |
| remap_fixed_half | 2.9 | 2.8 | 3.3 |

### 전체 합성 — 800x800 (5 cm/px)

| variant | median_ms | p10_ms | p90_ms |
|---|---|---|---|
| V1 | 31.9 | 27.7 | 35.2 |
| V1 +timecomp | 110.8 | 103.4 | 124.2 |
| V2 | 58.0 | 41.3 | 74.7 |
| V2 +timecomp | 522.4 | 167.9 | 914.6 |
| V3 | 158.7 | 142.8 | 207.9 |
| V3 +timecomp | 184.3 | 173.9 | 202.1 |
| V4 | 219.6 | 134.7 | 520.1 |
| V4 +timecomp | 155.4 | 152.2 | 443.9 |

### 전체 합성 — 400x400 (10 cm/px)

| variant | median_ms | p10_ms | p90_ms |
|---|---|---|---|
| V1 | 8.9 | 8.3 | 9.4 |
| V1 +timecomp | 16.9 | 16.1 | 18.4 |
| V2 | 9.1 | 8.2 | 9.9 |
| V2 +timecomp | 17.4 | 15.9 | 20.0 |
| V3 | 30.5 | 29.1 | 35.3 |
| V3 +timecomp | 37.0 | 36.3 | 40.9 |
| V4 | 32.4 | 31.6 | 35.7 |
| V4 +timecomp | 42.4 | 38.4 | 44.6 |
