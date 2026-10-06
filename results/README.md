Outputs of the scripts land here:

- `index.csv` / `dataset_stats.csv` — split identifier and class distribution (Table 1b)
- `graph_stats.json` — nodes, edges, degree distribution (Sec. 4.2)
- `tuned.json` — validation-selected hyper-parameters per method (Table 2)
- `tables.md` — Tables 3–12 in the manuscript's format
- `records/<tag>_seed<k>.json` — per-node predictions (`_p` = MC-dropout mean probability, `_y` labels), confusion counts, bootstrap intervals, training history
- `records/seed_<k>.json` — minimal records (`y_true`, `y_pred`, optionally `mu`) consumed by `scripts/report_from_records.py`
