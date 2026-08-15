# Kaggle semantic-adapter run

This private Kaggle script freezes `openai/clip-vit-base-patch32` and trains only a
1024-to-16-to-4 bottleneck classifier. It predicts one of four task intents from the
initial table image and language instruction. The exported `.npz` is evaluated later
in the local MuJoCo structured-waypoint executor; this remote run does not claim an
end-to-end action policy or OpenVLA LoRA fine-tuning.
