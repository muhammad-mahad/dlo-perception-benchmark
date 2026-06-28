# Benchmark report — phase1v4 Reproduce-and-Compare

Runs: batch1_gpu, batch1_cpu_cpu

## 3D tracking (Row 1)

| method | RMSE-3D mean (mm) | RMSE-3D med (mm) | len-err | self-int | FPS med | p95 lat (ms) | GPU | offline | errors |
|---|---|---|---|---|---|---|---|---|---|
| deform [cpu] | 140.3 | 128.0 | -0.191 | 0.91 | 14.8 | 153.8 | n | n | 0 |
| deform | 140.3 | 128.0 | -0.191 | 0.91 | 7.2 | 309.5 | n | n | 0 |
| der_ekf | 145.9 | 135.8 | -0.184 | 3.05 | 4.5 | 253.5 | n | n | 0 |
| infofusion | 146.0 | 135.8 | -0.186 | 3.10 | 21.7 | 62.9 | n | n | 0 |
| der_ukf | 146.1 | 136.0 | -0.181 | 3.03 | 14.0 | 94.1 | n | n | 0 |
| rts_smoother | 147.6 | 139.8 | -0.225 | 3.66 | 21.8 | 68.7 | n | Y | 0 |
| inextensible | 364.7 | 366.5 | 1.872 | 4.47 | 20.3 | 70.8 | n | n | 0 |
| particle_mjstep | 397.1 | 434.9 | -0.007 | 2.47 | 22.8 | 55.6 | n | n | 0 |
| inhouse_pipeline | 407.8 | 411.7 | 4.492 | 10.83 | 28.3 | 47.6 | n | n | 0 |

## 2D perception (Row 0)

| method | RMSE-2D (px) | chamfer (px) | mask IoU | FPS med | p95 lat (ms) | GPU | errors |
|---|---|---|---|---|---|---|---|
| fastdlo | 13.4 | 3.5 | 0.870 | 52.6 | 23.0 | n | 0 |
| fastdlo [cpu] | 13.4 | 3.5 | 0.870 | 50.6 | 25.5 | n | 0 |
| mbest | 14.2 | 3.1 | 0.870 | 64.3 | 20.6 | n | 0 |
| mbest [cpu] | 14.2 | 3.1 | 0.870 | 57.2 | 24.9 | n | 0 |
| rtdlo | 19.8 | 4.2 | 0.870 | 98.1 | 14.4 | n | 0 |
| rtdlo [cpu] | 20.3 | 4.2 | 0.870 | 84.8 | 14.3 | n | 0 |
| rtdlo_nativeseg | 32.4 | 7.3 | 0.377 | 27.5 | 37.9 | Y | 0 |
| rtdlo_nativeseg [cpu] | 32.5 | 7.1 | 0.377 | 1.5 | 766.8 | n | 0 |
| fastdlo_nativeseg [cpu] | 34.3 | 14.0 | 0.379 | 1.4 | 850.8 | n | 0 |
| fastdlo_nativeseg | 34.3 | 14.0 | 0.378 | 18.8 | 59.7 | Y | 0 |

## Per-scenario breakdown

### 3D RMSE (mm) per scenario

| method | s1_free_hang | s2_forced_bend | s3_twist | s4_buckling | s5_tangling | s6_snap | s7_plastic |
|---|---|---|---|---|---|---|---|
| deform | 131.0 | 130.2 | 139.1 | 181.3 | 114.3 | 146.5 | 139.5 |
| deform [cpu] | 131.0 | 130.2 | 139.1 | 181.3 | 114.3 | 146.5 | 139.5 |
| der_ekf | 130.9 | 146.5 | 142.2 | 174.4 | 121.8 | 158.2 | 147.5 |
| der_ukf | 130.9 | 146.5 | 142.3 | 174.6 | 122.0 | 158.8 | 147.7 |
| inextensible | 138.4 | 347.9 | 482.5 | 291.1 | 534.7 | 344.7 | 413.7 |
| infofusion | 131.1 | 146.4 | 142.3 | 174.4 | 121.9 | 158.2 | 147.4 |
| inhouse_pipeline | 142.1 | 393.7 | 522.5 | 327.3 | 614.3 | 388.9 | 466.0 |
| particle_mjstep | 491.4 | 351.6 | 409.2 | 290.2 | 483.6 | 341.0 | 412.4 |
| rts_smoother | 130.5 | 153.7 | 141.6 | 175.3 | 124.7 | 160.3 | 147.4 |

### 2D RMSE (px) per scenario

| method | s1_free_hang | s2_forced_bend | s3_twist | s4_buckling | s5_tangling | s6_snap | s7_plastic |
|---|---|---|---|---|---|---|---|
| fastdlo | 39.6 | 3.2 | 5.9 | 11.4 | 21.3 | 6.5 | 6.0 |
| fastdlo [cpu] | 39.6 | 3.2 | 5.9 | 11.4 | 21.3 | 6.5 | 6.0 |
| fastdlo_nativeseg | 54.9 | 29.9 | 33.7 | 28.2 | 47.0 | 22.9 | 23.7 |
| fastdlo_nativeseg [cpu] | 54.9 | 29.9 | 33.8 | 27.8 | 46.9 | 23.0 | 23.7 |
| mbest | 38.4 | 5.2 | 8.9 | 11.6 | 20.9 | 7.2 | 7.3 |
| mbest [cpu] | 38.4 | 5.2 | 8.9 | 11.6 | 20.9 | 7.2 | 7.3 |
| rtdlo | 41.0 | 4.6 | 16.4 | 16.2 | 25.2 | 21.8 | 13.4 |
| rtdlo [cpu] | 41.1 | 4.2 | 18.4 | 16.4 | 25.0 | 24.6 | 12.6 |
| rtdlo_nativeseg | 48.7 | 32.8 | 26.6 | 29.7 | 34.3 | 32.4 | 22.3 |
| rtdlo_nativeseg [cpu] | 49.7 | 31.5 | 27.2 | 28.6 | 33.4 | 32.5 | 24.6 |

## Frontier gap

Best real-time 3D tracker: **deform [cpu]** at 140.3 mm / 14.8 FPS. The empty frontier region (real-time AND better accuracy than this point, at low GPU cost) is the gap the novel method targets.
