# Point-Match Benchmark: pm_baseline

**500 images, 746 GT boxes**

## Configuration
- Snap window: 20px
- Snap passes: 2
- YOLO conf: 0.55
- Edge sampling: full (0.0-1.0)
- Line fit: Huber (nearest red per profile)

## Results

| Method | D=3 | D=5 | D=8 | D=12 |
|--------|------|------|------|------|
| m8 | 93 | 176 | 235 | 284 |
| YOLO | 12 | 80 | 201 | 321 |
| YOLO+snap | 70 | 209 | 275 | 330 |
| combine+snap | 120 | 243 | 298 | 355 |

## Recall (%)

| Method | D=3 | D=5 | D=8 | D=12 |
|--------|------|------|------|------|
| m8 | 12.5% | 23.6% | 31.5% | 38.1% |
| YOLO | 1.6% | 10.7% | 26.9% | 43.0% |
| YOLO+snap | 9.4% | 28.0% | 36.9% | 44.2% |
| combine+snap | 16.1% | 32.6% | 39.9% | 47.6% |

## Key Findings

- **m8 corners are already on the red mask (92.9%)** — snapping m8 alone hurts
- **GT corners are only on the red mask 76.9%** — fundamental precision limit
- **YOLO+snap improves YOLO by 2.6x** at D=5 (80→209 TP)
- **combine+snap is best at all D** — union of m8 precision + YOLO recall
