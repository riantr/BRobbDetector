# DoubleML Error Cause Analysis (20261010)

## Error Summary

- Error images: 146 / 500
- FN-only images: 17, FP-only: 82, Both: 47
- Total FN boxes: 69, Total FP boxes: 238
- Data copied to: `E:\Xray\CARS\data\Image@hard_20261010`, `E:\Xray\CARS\data\Label@hard_20261010`, `E:\Xray\CARS\data\Det@hard_20261010`

## DoubleML Causal Models

| Model | Y | D | Type | theta | SE | t-stat | p-value | 95% CI | Significant? |
|-------|---|---|------|-------|-----|--------|---------|--------|-------------|
| M1_gt_incomplete_on_fp | fp | gt_incomplete | PLR | -0.4540 | 0.3638 | -1.25 | 0.2120 | [-1.1670, 0.2590] | no |
| M2_gt_incomplete_on_has_fp | has_fp | gt_incomplete | IRM | 0.6124 | 0.0152 | 40.27 | 0.0000 | [0.5826, 0.6423] | **YES** |
| M3_gt_avg_aspect_on_fn | fn | gt_avg_aspect | PLR | -0.0365 | 0.0318 | -1.15 | 0.2513 | [-0.0988, 0.0258] | no |
| M4_gt_has_strip_on_fn | fn | gt_has_strip | PLR | -0.1093 | 0.1238 | -0.88 | 0.3770 | [-0.3519, 0.1333] | no |
| M5_red_density_on_fn | fn | red_density | PLR | 0.4006 | 2.0469 | 0.20 | 0.8448 | [-3.6113, 4.4125] | no |
| M6_red_density_on_fp | fp | red_density | PLR | 3.4580 | 5.1179 | 0.68 | 0.4992 | [-6.5731, 13.4891] | no |
| M7_n_red_components_on_fp | fp | n_red_components | PLR | -0.0005 | 0.0003 | -1.74 | 0.0810 | [-0.0011, 0.0001] | no |
| M8_gt_box_density_on_fn | fn | gt_box_density | PLR | 113581.3453 | 30532.4966 | 3.72 | 0.0002 | [53737.6519, 173425.0387] | **YES** |
| M9_gt_incomplete_on_has_error | has_error | gt_incomplete | IRM | 0.5800 | 0.0172 | 33.70 | 0.0000 | [0.5463, 0.6138] | **YES** |
| M10_gt_avg_min_side_on_fn | fn | gt_avg_min_side | PLR | 0.0011 | 0.0013 | 0.86 | 0.3921 | [-0.0014, 0.0035] | no |

### Interpretation

- **M1_gt_incomplete_on_fp** (GT incompleteness -> FP count): theta=-0.4540, p=0.2120 (not significant). Effect is negative: 
  no clear causal effect.
- **M2_gt_incomplete_on_has_fp** (GT incompleteness -> has_any_FP): theta=0.6124, p=0.0000 (statistically significant). Effect is positive: 
  increasing gt_incomplete increases has_fp.
- **M3_gt_avg_aspect_on_fn** (GT avg aspect ratio -> FN count): theta=-0.0365, p=0.2513 (not significant). Effect is negative: 
  no clear causal effect.
- **M4_gt_has_strip_on_fn** (GT has strip-like box -> FN count): theta=-0.1093, p=0.3770 (not significant). Effect is negative: 
  no clear causal effect.
- **M5_red_density_on_fn** (Red pixel density -> FN count): theta=0.4006, p=0.8448 (not significant). Effect is positive: 
  no clear causal effect.
- **M6_red_density_on_fp** (Red pixel density -> FP count): theta=3.4580, p=0.4992 (not significant). Effect is positive: 
  no clear causal effect.
- **M7_n_red_components_on_fp** (N red components -> FP count): theta=-0.0005, p=0.0810 (not significant). Effect is negative: 
  no clear causal effect.
- **M8_gt_box_density_on_fn** (GT box density -> FN count): theta=113581.3453, p=0.0002 (statistically significant). Effect is positive: 
  increasing gt_box_density increases fn.
- **M9_gt_incomplete_on_has_error** (GT incompleteness -> has_any_error): theta=0.5800, p=0.0000 (statistically significant). Effect is positive: 
  increasing gt_incomplete increases has_error.
- **M10_gt_avg_min_side_on_fn** (GT avg min side -> FN count): theta=0.0011, p=0.3921 (not significant). Effect is positive: 
  no clear causal effect.

## RandomForest Feature Importance

### Predicting fp

| Feature | Importance |
|---------|-----------|
| csv_gt_gap | 0.4632 |
| gt_incomplete | 0.0852 |
| gt_avg_area | 0.0562 |
| img_area | 0.0536 |
| gt_avg_max_side | 0.0390 |
| img_height | 0.0381 |
| contrast | 0.0285 |
| red_pixel_count | 0.0277 |
| brightness | 0.0275 |
| gt_avg_min_side | 0.0261 |
| n_red_components | 0.0233 |
| largest_comp_frac | 0.0226 |
| red_density | 0.0215 |
| img_width | 0.0183 |
| gt_avg_aspect | 0.0160 |

### Predicting fn

| Feature | Importance |
|---------|-----------|
| gt_avg_area | 0.1087 |
| csv_gt_gap | 0.0918 |
| red_pixel_count | 0.0782 |
| gt_avg_min_side | 0.0563 |
| gt_avg_max_side | 0.0556 |
| brightness | 0.0549 |
| contrast | 0.0543 |
| gt_avg_aspect | 0.0501 |
| gt_spread_y | 0.0468 |
| red_density | 0.0459 |
| largest_comp_frac | 0.0426 |
| img_height | 0.0406 |
| n_red_components | 0.0402 |
| gt_box_density | 0.0376 |
| gt_std_aspect | 0.0354 |

### Predicting has_error

| Feature | Importance |
|---------|-----------|
| csv_gt_gap | 0.3245 |
| gt_incomplete | 0.2540 |
| img_height | 0.0463 |
| n_red_components | 0.0404 |
| red_pixel_count | 0.0390 |
| brightness | 0.0359 |
| contrast | 0.0314 |
| gt_avg_min_side | 0.0284 |
| gt_avg_aspect | 0.0255 |
| img_width | 0.0223 |
| img_area | 0.0197 |
| red_density | 0.0196 |
| gt_avg_max_side | 0.0194 |
| largest_comp_frac | 0.0187 |
| gt_avg_area | 0.0168 |

## Conclusion

Significant causal factors (3):
- **gt_incomplete** -> has_fp: theta=0.6124 (p=0.0000)
- **gt_box_density** -> fn: theta=113581.3453 (p=0.0002)
- **gt_incomplete** -> has_error: theta=0.5800 (p=0.0000)