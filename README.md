
---

## 1. 环境要求

- Python 3.9 – 3.12


```bash
pip install -r requirements.txt
```

说明：
- 所有测试图均来自 scikit-image 自带数据集。

---

## 2. 目录结构

```
cair_project/
├── README.md
├── requirements.txt
├── cair/                 # 算法包
│   ├── __init__.py       # 公共 API
│   ├── energy.py         # 梯度能量 / 显著性 / 前景掩膜 / 重要度图(式1)
│   ├── seam.py           # 接缝裁剪 DP、增删接缝、经典 SC 基线
│   ├── composition.py    # 四条构图规则 + 语义线检测 + 统一入口 resize()
│   ├── metric.py         # 质量指数（信息损失 + 几何形变）
│   └── baselines.py      # 美学裁剪基线 [22]
├── make_inputs.py        # 构建测试图集（写入 outputs/inputs/）
├── run_experiments.py    # 复现 Table 1 与 Fig.1–4 四联对比图
└── run_scalability.py    # 复现 O(W·H·k) 可扩展性实验
```

---

## 3. 复现实验

```bash
cd cair_project

python3 run_experiments.py
python3 run_scalability.py
```
