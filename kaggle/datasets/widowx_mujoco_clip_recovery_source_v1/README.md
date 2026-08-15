# WidowX MuJoCo CLIP Recovery Source v1

Private Kaggle source for a frozen-CLIP recovery-value adapter.

- 52 post-failure MuJoCo RGB states
- 41 train / 11 seed-disjoint test samples
- labels: `stop` versus `retry`, obtained by executing one counterfactual RGB-relocalized recovery trajectory
- both top and front RGB views are stored, but top-only and top+front are evaluated as separate ablations

This is a MuJoCo-only study. The source contains no real-robot data and no simulator state is used as a runtime model input.
