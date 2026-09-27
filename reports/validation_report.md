# Data validation report

Generated (UTC): 2026-09-27T18:07:11+00:00

| Table | Rows in | Rows out |
|---|---:|---:|
| telemetry | 864,000 | 864,000 |
| failures | 9 | 9 |
| maintenance | 51 | 51 |

| Table | Check | Severity | Failed | Action | Detail |
|---|---|---|---:|---|---|
| telemetry | unique (machine_id, timestamp) | info | 0 | pass |  |
| telemetry | contract schema | info | 0 | pass |  |
| telemetry | pressure_bar within [0, 350] | warning | 250 | set NaN in cleaning |  |
| telemetry | temp_celsius within [10, 90] | warning | 170 | set NaN in cleaning |  |
| telemetry | flow_lpm within [0, 150] | info | 0 | pass |  |
| telemetry | vibration_x_g within [0, 16] | info | 0 | pass |  |
| telemetry | vibration_y_g within [0, 16] | info | 0 | pass |  |
| telemetry | pump_rpm within [500, 1600] | info | 0 | pass |  |
| telemetry | vibration_x_g readings exactly at sensor floor 0.05 | warning | 206,697 | keep; exclude floor-driven minima from features | 23.9% of rows |
| telemetry | vibration_y_g readings exactly at sensor floor 0.05 | warning | 209,734 | keep; exclude floor-driven minima from features | 24.3% of rows |
| telemetry | missing sensors ⇔ dropout flag | info | 0 | pass | 2,590 dropout rows |
| telemetry | regular 1min grid | info | 0 | pass | 0 off-grid timestamps |
| telemetry | label columns present in telemetry | info | 3 | excluded from features | is_anomaly, failure_mode, rul_hours |
| failures | contract schema | info | 0 | pass |  |
| maintenance | contract schema | info | 0 | pass |  |
| failures | machine_id exists in telemetry | info | 0 | pass |  |
| maintenance | machine_id exists in telemetry | info | 0 | pass |  |
| failures | events inside telemetry period | info | 0 | report only | 9 of 9 inside 2024-01-01–2024-02-29 |
| maintenance | events inside telemetry period | warning | 51 | report only | 0 of 51 inside 2024-01-01–2024-02-29 |
