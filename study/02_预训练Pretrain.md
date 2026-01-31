# 02 - 预训练 Pretrain 详解

## 核心文件
- `trainer/train_pretrain.py` (166行)
- `dataset/lm_dataset.py` (PretrainDataset)
- `trainer/trainer_utils.py` (工具函数)

---

## 一、预训练的原理

### 1.1 目标：Next Token Prediction

预训练的核心任务是**预测下一个token**：

```
输入: "今天天气"  → token_ids: [100, 200, 200, 300]
标签: "天天气真"  → token_ids: [200, 200, 300, 400]

模型需要学会：
- 给定 "今"，预测 "天"
- 给定 "今天"，预测 "天"
- 给定 "今天天"，预测 "气"
- 给定 "今天天气"，预测 "真"
```

### 1.2 为什么有效

通过海量文本的预测任务，模型学会了：
- **语法**：主谓宾结构、时态变化
- **语义**：词语之间的关联
- **常识**：太阳从东边升起
- **推理**：如果A则B的逻辑

---

## 二、命令行参数详解

```python
parser.add_argument("--save_dir", default="../out")
parser.add_argument('--save_weight', default='pretrain')
parser.add_argument("--epochs", default=1)
parser.add_argument("--batch_size", default=32)
parser.add_argument("--learning_rate", default=5e-4)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--dtype", default="bfloat16")
parser.add_argument("--num_workers", default=8)
parser.add_argument("--accumulation_steps", default=8)
parser.add_argument("--grad_clip", default=1.0)
parser.add_argument("--log_interval", default=100)
parser.add_argument("--save_interval", default=1000)
parser.add_argument('--hidden_size', default=512)
parser.add_argument('--num_hidden_layers', default=8)
parser.add_argument('--max_seq_len', default=340)
parser.add_argument('--use_moe', default=0)
parser.add_argument("--data_path", default="../dataset/pretrain_hq.jsonl")
parser.add_argument('--from_weight', default='none')
parser.add_argument('--from_resume', default=0)
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--use_compile", default=0)
```

### 参数详细说明

| 参数 | 默认值 | 作用 | 建议值 |
|------|--------|------|--------|
| `--epochs` | 1 | 训练轮数 | 1-2轮（预训练数据大）|
| `--batch_size` | 32 | 每批样本数 | 根据显存调整 |
| `--learning_rate` | 5e-4 | 初始学习率 | 预训练用较大lr |
| `--accumulation_steps` | 8 | 梯度累积步数 | 等效batch=32*8=256 |
| `--grad_clip` | 1.0 | 梯度裁剪阈值 | 防止梯度爆炸 |
| `--max_seq_len` | 340 | 最大序列长度 | 中文约500字 |
| `--hidden_size` | 512 | 模型隐藏层大小 | 512(Small)/768(Base) |
| `--num_hidden_layers` | 8 | Transformer层数 | 8(Small)/16(Base) |
| `--use_moe` | 0 | 是否用MoE架构 | 0=否，1=是 |
| `--dtype` | bfloat16 | 混合精度类型 | bfloat16更稳定 |
| `--from_weight` | none | 基于哪个权重 | none=从头训练 |
| `--from_resume` | 0 | 是否断点续训 | 1=自动检测续训 |

---

## 三、训练流程详解

### 3.1 初始化阶段 (第99-131行)

```python
# ========== 1. 初始化环境和随机种子 ==========
local_rank = init_distributed_mode()  # 初始化分布式（如果有）
if dist.is_initialized():
    args.device = f"cuda:{local_rank}"
setup_seed(42 + rank)  # 设置随机种子，保证可复现

# ========== 2. 配置目录、模型参数、检查ckp ==========
os.makedirs(args.save_dir, exist_ok=True)
lm_config = MiniMindConfig(
    hidden_size=args.hidden_size,       # 512
    num_hidden_layers=args.num_hidden_layers,  # 8
    use_moe=bool(args.use_moe)          # False
)
# 如果from_resume=1，尝试加载断点
ckp_data = lm_checkpoint(...) if args.from_resume==1 else None

# ========== 3. 设置混合精度 ==========
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)

# ========== 5. 定义模型、数据、优化器 ==========
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
```

### 3.2 数据集 PretrainDataset

**数据格式** (jsonl文件)：
```json
{"text": "这是第一段预训练文本，包含各种知识..."}
{"text": "这是第二段预训练文本，包含更多内容..."}
```

**数据集实现**：
```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=340):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载jsonl文件
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        # 分词
        tokens = self.tokenizer.encode(text)
        # 添加BOS和EOS
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 截断或填充到max_length
        tokens = tokens[:self.max_length]
        tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        input_ids = torch.tensor(tokens)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # pad位置不计算loss

        return input_ids, labels
```

### 3.3 训练循环 train_epoch (第23-71行)

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 数据移到GPU
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 动态调整学习率（余弦退火）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（混合精度）
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss  # aux_loss是MoE的辅助损失
            loss = loss / args.accumulation_steps  # 梯度累积

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积完成后更新参数
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志打印
        if step % args.log_interval == 0:
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), loss: {loss:.4f}, lr: {lr:.8f}')

        # 保存模型
        if step % args.save_interval == 0 and is_main_process():
            torch.save(model.state_dict(), f'{args.save_dir}/pretrain_{hidden_size}.pth')
```

---

## 四、关键技术详解

### 4.1 学习率调度 get_lr

```python
def get_lr(current_step, total_steps, lr):
    # 余弦退火：从lr逐渐降到0.1*lr
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

**图示**：
```
lr
 ^
 |  ___
 | /   \
 |/     \___
 +-----------> step
   warm   decay
```

**为什么用余弦退火**：
- 开始学习率高，快速学习
- 后期学习率低，精细调整
- 比固定lr效果好

### 4.2 混合精度训练

```python
# 自动混合精度上下文
autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)

# GradScaler（仅float16需要）
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

# 使用
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**bfloat16 vs float16**：
| 类型 | 精度 | 动态范围 | 是否需要Scaler |
|------|------|----------|----------------|
| float16 | 高 | 小 | 需要 |
| bfloat16 | 低 | 大 | 不需要 |

**建议**：优先用bfloat16，更稳定不会溢出

### 4.3 梯度累积

```python
loss = loss / args.accumulation_steps  # 8

scaler.scale(loss).backward()

if (step + 1) % args.accumulation_steps == 0:
    scaler.step(optimizer)  # 每8步才更新一次参数
    optimizer.zero_grad()
```

**作用**：显存不够时，模拟更大的batch_size

```
实际batch_size = 32
accumulation_steps = 8
等效batch_size = 32 * 8 = 256
```

### 4.4 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

**作用**：防止梯度爆炸

```
如果 ||grad|| > 1.0:
    grad = grad * (1.0 / ||grad||)
```

### 4.5 断点续训

```python
# 保存断点
lm_checkpoint(lm_config, weight='pretrain', model=model, optimizer=optimizer,
              scaler=scaler, epoch=epoch, step=step, wandb=wandb)

# 加载断点
ckp_data = lm_checkpoint(lm_config, weight='pretrain')
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    scaler.load_state_dict(ckp_data['scaler'])
    start_epoch = ckp_data['epoch']
    start_step = ckp_data['step']
```

---

## 五、分布式训练 (DDP)

### 5.1 初始化

```python
local_rank = init_distributed_mode()  # 获取当前GPU编号

def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 单卡模式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
```

### 5.2 包装模型

```python
if dist.is_initialized():
    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
    model = DistributedDataParallel(model, device_ids=[local_rank])
```

### 5.3 数据采样

```python
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
loader = DataLoader(train_ds, batch_sampler=batch_sampler, ...)
```

### 5.4 启动多卡训练

```bash
# 单卡
python trainer/train_pretrain.py

# 多卡 (4张GPU)
torchrun --nproc_per_node=4 trainer/train_pretrain.py
```

---

## 六、运行示例

### 6.1 基本运行（单卡）

```bash
python trainer/train_pretrain.py \
    --epochs 1 \
    --batch_size 64 \
    --learning_rate 5e-4 \
    --max_seq_len 340 \
    --data_path ../dataset/pretrain_hq.jsonl
```

### 6.2 多卡训练（4 × RTX 4090）

```bash
# 使用 torchrun 启动 4 卡训练
torchrun --nproc_per_node=4 trainer/train_pretrain.py \
    --epochs 1 \
    --batch_size 64 \
    --learning_rate 5e-4

# 等效 batch_size = 64 × 4 = 256
```

### 6.3 小规模测试

```bash
python trainer/train_pretrain.py \
    --epochs 1 \
    --batch_size 8 \
    --save_interval 100 \
    --log_interval 10
```

### 6.4 断点续训

```bash
python trainer/train_pretrain.py \
    --from_resume 1
```

### 6.5 使用MoE

```bash
python trainer/train_pretrain.py \
    --use_moe 1 \
    --hidden_size 640
```

### 6.6 RTX 4090 参数建议

| 参数 | 单卡建议 | 4卡建议 |
|------|----------|---------|
| batch_size | 64-128 | 64 (等效256) |
| accumulation_steps | 4 | 2 |
| 显存占用 | ~8GB | ~8GB/卡 |
| 预计时间 (1 epoch) | ~15分钟 | ~4分钟 |

---

## 七、输出文件

训练完成后会生成：

```
out/
├── pretrain_512.pth          # 模型权重（用于推理）
└── checkpoints/
    └── pretrain_512_resume.pth  # 完整断点（用于续训）
```

---

## 八、常见问题

### Q1: 显存不够怎么办？
- 减小 `batch_size`
- 增大 `accumulation_steps`
- 减小 `max_seq_len`

### Q2: 训练太慢？
- 增大 `batch_size`
- 使用 `--use_compile 1`（PyTorch 2.0+）
- 多卡训练

### Q3: Loss不下降？
- 检查数据格式是否正确
- 尝试减小学习率
- 检查是否有NaN
