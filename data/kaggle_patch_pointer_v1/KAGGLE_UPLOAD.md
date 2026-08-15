# Kaggle 上传与训练

1. 上传本目录或同名 ZIP 为 Kaggle Dataset，建议名称 `widowx-mujoco-patch-pointer-v1`。
2. 创建 Kaggle Notebook，开启 GPU 和 Internet，并附加这个 Dataset。
3. 上传或打开 `kaggle_train_patch_pointer.ipynb`，确认输入目录为 `/kaggle/input/widowx-mujoco-patch-pointer-v1`，运行训练。
4. 下载 `clip_patch_pointer_kaggle_v1.pt` 与 `clip_patch_pointer_kaggle_v1.json` 到本机后，使用固定 MuJoCo 留出 `20-24/120-124/220-224/420-424` 运行 `scripts/evaluate_clip_patch_pointer.py`。

该训练包只对应冻结 CLIP patch-token 空间指针头。它不是 OpenVLA LoRA 运行包，不能以其训练结果宣称完成 OpenVLA、VLA foundation-model 或真实机械臂实验。
