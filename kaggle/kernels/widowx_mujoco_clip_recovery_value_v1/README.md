# Kaggle CLIP recovery-value run

The kernel freezes `openai/clip-vit-base-patch32` and trains only an 8-dimensional
bottleneck classifier for `stop` versus `retry` on real MuJoCo post-failure RGB
states. It exports top-only and top+front `.npz` heads for local MuJoCo evaluation.

The head is queried only after the local RGB terminal check reports incomplete and
RGB re-localization still finds the source object. It is not an action policy,
OpenVLA LoRA, or real-robot result.
