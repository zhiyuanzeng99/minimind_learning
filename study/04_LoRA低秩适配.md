# 04 - LoRA 低秩适配详解

## 核心文件
- `model/model_lora.py` (54行) - **必看，核心实现**
- `trainer/train_lora.py` (174行)

---

## 一、LoRA的原理

### 1.1 问题：全参数微调的缺点

全参数SFT的问题：
1. **显存占用大**：需要存储所有参数的梯度
2. **训练慢**：所有参数都要更新
3. **无法多任务**：每个任务需要保存完整模型副本

### 1.2 LoRA的核心思想

**Low-Rank Adaptation (低秩适配)**

核心观察：微调时，权重的变化量ΔW是**低秩**的

```
原始：Y = W × X              W是512×512的大矩阵

微调后：Y = (W + ΔW) × X     ΔW也是512×512

LoRA的关键洞察：ΔW可以分解为两个小矩阵的乘积
ΔW ≈ B × A                   A是512×8，B是8×512

所以：Y = W × X + B × A × X
```

### 1.3 参数量对比

```
原始矩阵W: 512 × 512 = 262,144 参数
LoRA矩阵: 512 × 8 + 8 × 512 = 8,192 参数

参数减少: 262,144 / 8,192 = 32倍
```

### 1.4 为什么有效

微调时，权重变化主要集中在一个低维子空间：
- 大部分参数变化很小
- 只有少数"方向"的变化显著
- 用低秩矩阵就能捕捉这些主要变化

---

## 二、代码实现详解

### 2.1 LoRA类 (model_lora.py 第6-18行)

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩，通常是8

        # 低秩矩阵A: in_features → rank (降维)
        self.A = nn.Linear(in_features, rank, bias=False)

        # 低秩矩阵B: rank → out_features (升维)
        self.B = nn.Linear(rank, out_features, bias=False)

        # 初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)  # A高斯初始化
        self.B.weight.data.zero_()  # B全零初始化（确保初始时ΔW=0）

    def forward(self, x):
        # ΔW × x = B(A(x))
        return self.B(self.A(x))
```

**初始化策略**：
- A用高斯初始化
- B用零初始化
- 这样开始时 `B×A = 0`，模型行为与原始相同

### 2.2 应用LoRA (第21-32行)

```python
def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        # 只对方阵Linear层添加LoRA
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建LoRA模块
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank)
            lora = lora.to(model.device)

            # 将LoRA附加到原模块
            setattr(module, "lora", lora)

            # 修改forward函数
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)  # 原始输出 + LoRA输出

            module.forward = forward_with_lora
```

**为什么只对方阵**：
- 方阵（如q_proj, k_proj, v_proj, o_proj）是注意力的核心
- 这些层对微调效果影响最大
- 非方阵层（如embed, lm_head）不添加LoRA

### 2.3 保存LoRA (第45-53行)

```python
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}

    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 只保存lora的A和B矩阵
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)

    torch.save(state_dict, path)
```

**保存的内容**：
```python
{
    'model.layers.0.self_attn.q_proj.lora.A.weight': tensor(...),
    'model.layers.0.self_attn.q_proj.lora.B.weight': tensor(...),
    'model.layers.0.self_attn.k_proj.lora.A.weight': tensor(...),
    ...
}
```

### 2.4 加载LoRA (第35-42行)

```python
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    # 去掉可能的'module.'前缀
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 找到对应的lora权重并加载
            lora_state = {k.replace(f'{name}.lora.', ''): v
                         for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)
```

---

## 三、训练脚本详解

### 3.1 关键参数

```python
parser.add_argument('--lora_name', default='lora_identity')  # LoRA权重名
parser.add_argument("--learning_rate", default=1e-4)         # 比SFT大，因为只训练LoRA
parser.add_argument("--epochs", default=50)                  # LoRA可以训练更多轮
parser.add_argument('--from_weight', default='pretrain')     # 基于预训练/SFT权重
```

### 3.2 核心训练流程

```python
# 1. 加载基础模型
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

# 2. 应用LoRA
from model.model_lora import apply_lora, save_lora
apply_lora(model, rank=8)

# 3. 冻结原始参数，只训练LoRA参数
for name, param in model.named_parameters():
    if 'lora' not in name:
        param.requires_grad = False  # 冻结非LoRA参数

# 4. 只优化LoRA参数
lora_params = [p for n, p in model.named_parameters() if 'lora' in n and p.requires_grad]
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

# 5. 训练循环（与SFT相同）
for epoch in range(args.epochs):
    for input_ids, labels in loader:
        res = model(input_ids, labels=labels)
        loss = res.loss
        loss.backward()
        optimizer.step()

# 6. 保存LoRA权重（只保存LoRA部分）
save_lora(model, f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth')
```

---

## 四、LoRA vs 全参数微调

| 对比项 | 全参数SFT | LoRA |
|--------|-----------|------|
| **训练参数量** | 26M (100%) | ~1M (4%) |
| **显存占用** | 高 | 低 |
| **训练速度** | 慢 | 快 |
| **保存大小** | ~100MB | ~4MB |
| **学习率** | 1e-6 | 1e-4 |
| **训练轮数** | 2-3轮 | 10-50轮 |
| **多任务** | 每任务一个模型 | 每任务一个LoRA文件 |
| **效果** | 略好 | 接近 |

---

## 五、运行示例

### 5.1 训练身份识别LoRA

```bash
python trainer/train_lora.py \
    --lora_name lora_identity \
    --epochs 50 \
    --learning_rate 1e-4 \
    --from_weight full_sft \
    --data_path ../dataset/lora_identity.jsonl
```

### 5.2 训练医疗领域LoRA

```bash
python trainer/train_lora.py \
    --lora_name lora_medical \
    --epochs 30 \
    --data_path ../dataset/lora_medical.jsonl
```

### 5.3 使用LoRA推理

```bash
python eval_llm.py \
    --weight full_sft \
    --lora_weight lora_identity
```

---

## 六、LoRA数据准备

### 6.1 身份识别数据示例

```json
{"conversations": [{"role": "user", "content": "你是谁？"}, {"role": "assistant", "content": "我是MiniMind，一个由xxx开发的AI助手。"}]}
{"conversations": [{"role": "user", "content": "你叫什么名字？"}, {"role": "assistant", "content": "我叫MiniMind，很高兴认识你！"}]}
```

### 6.2 领域知识数据示例

```json
{"conversations": [{"role": "user", "content": "什么是高血压？"}, {"role": "assistant", "content": "高血压是指血压持续高于正常值的疾病..."}]}
```

---

## 七、高级用法

### 7.1 合并多个LoRA

```python
# 加载基础模型
model, _ = init_model(config, 'full_sft')
apply_lora(model, rank=8)

# 加载LoRA 1
load_lora(model, 'lora_identity.pth')
# 此时模型有了身份识别能力

# 也可以加载LoRA 2（会覆盖）
load_lora(model, 'lora_medical.pth')
# 此时模型有了医疗知识
```

### 7.2 LoRA权重融合

训练完成后，可以把LoRA权重合并到原始权重中：

```python
def merge_lora(model):
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 计算 ΔW = B × A
            delta_w = module.lora.B.weight @ module.lora.A.weight
            # 合并到原始权重
            module.weight.data += delta_w
            # 删除LoRA
            delattr(module, 'lora')
```

---

## 八、调参建议

### 8.1 rank的选择

```
rank=4:  参数最少，效果可能不够
rank=8:  推荐默认值，平衡效果和效率
rank=16: 复杂任务可以尝试
rank=32: 几乎等同于全参数微调
```

### 8.2 学习率

```
1e-4: 推荐起始值
1e-3: 数据量少时可以尝试
1e-5: 数据量大时可以尝试
```

### 8.3 训练轮数

```
数据量 < 1000: 50-100轮
数据量 1000-10000: 20-50轮
数据量 > 10000: 10-20轮
```

---

## 九、常见问题

### Q1: LoRA效果不好？
- 增大rank（8→16）
- 增加训练轮数
- 检查数据质量
- 尝试不同的学习率

### Q2: 训练不稳定？
- 减小学习率
- 增加warmup
- 检查是否有NaN

### Q3: 想训练更多层？
修改 `apply_lora` 函数，不限制只对方阵层添加LoRA

---

## 十、输出文件

```
out/
├── lora/
│   ├── lora_identity_512.pth   # 身份LoRA（很小，~4MB）
│   └── lora_medical_512.pth    # 医疗LoRA
└── full_sft_512.pth            # 基础模型（~100MB）
```

**优势**：可以用一个基础模型 + 多个LoRA文件实现多种能力
