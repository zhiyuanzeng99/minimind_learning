# 03 - SFT 监督微调详解

## 核心文件
- `trainer/train_full_sft.py` (161行)
- `dataset/lm_dataset.py` (SFTDataset)

---

## 一、SFT的原理

### 1.1 什么是SFT

**SFT (Supervised Fine-Tuning)** = 监督微调

预训练后的模型会"说话"，但不会"对话"：
- 预训练模型：输入"天气"，输出"预报显示明天..."（续写）
- SFT后模型：输入"今天天气怎么样？"，输出"今天天气晴朗，温度适宜。"（回答）

### 1.2 SFT vs Pretrain 对比

| 对比项 | Pretrain | SFT |
|--------|----------|-----|
| **目标** | 学习语言规律 | 学习对话能力 |
| **数据** | 纯文本 | 问答对 |
| **Loss计算** | 全部token | 只算回复部分 |
| **学习率** | 较大 (5e-4) | 较小 (1e-6) |
| **训练量** | 海量数据 | 精选数据 |
| **权重** | 随机初始化 | 加载预训练权重 |

### 1.3 为什么只对回复计算Loss

```
用户: 你好，请问今天天气怎么样？
助手: 今天天气晴朗，适合外出。

我们希望模型学会的是：
- ✓ 生成"今天天气晴朗，适合外出。"
- ✗ 生成"你好，请问今天天气怎么样？"（这是用户说的）

所以只对助手回复部分计算loss
```

---

## 二、与Pretrain的代码差异

### 2.1 主要差异点

```python
# train_pretrain.py
parser.add_argument('--save_weight', default='pretrain')
parser.add_argument("--learning_rate", default=5e-4)      # 较大
parser.add_argument("--data_path", default="../dataset/pretrain_hq.jsonl")
parser.add_argument('--from_weight', default='none')      # 从头训练
train_ds = PretrainDataset(...)                           # 预训练数据集

# train_full_sft.py
parser.add_argument('--save_weight', default='full_sft')
parser.add_argument("--learning_rate", default=1e-6)      # 较小，防止遗忘
parser.add_argument("--data_path", default="../dataset/sft_mini_512.jsonl")
parser.add_argument('--from_weight', default='pretrain')  # 加载预训练权重
train_ds = SFTDataset(...)                                # SFT数据集
```

### 2.2 关键差异：数据集

**Pretrain数据**：
```json
{"text": "这是一段纯文本..."}
```

**SFT数据**：
```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
  ]
}
```

---

## 三、命令行参数详解

```python
parser.add_argument("--save_dir", default="../out")
parser.add_argument('--save_weight', default='full_sft')
parser.add_argument("--epochs", default=2)           # SFT通常多轮
parser.add_argument("--batch_size", default=16)
parser.add_argument("--learning_rate", default=1e-6) # 关键：很小的lr
parser.add_argument("--accumulation_steps", default=1)
parser.add_argument('--max_seq_len', default=340)
parser.add_argument("--data_path", default="../dataset/sft_mini_512.jsonl")
parser.add_argument('--from_weight', default='pretrain')  # 加载预训练权重
```

### 参数对比

| 参数 | Pretrain | SFT | 原因 |
|------|----------|-----|------|
| `learning_rate` | 5e-4 | 1e-6 | SFT用小lr防止遗忘预训练知识 |
| `epochs` | 1 | 2 | SFT数据少，多轮学习 |
| `from_weight` | none | pretrain | SFT基于预训练模型 |
| `batch_size` | 32 | 16 | SFT对话较长，显存占用大 |
| `accumulation_steps` | 8 | 1 | SFT不需要太大的等效batch |

---

## 四、SFTDataset 详解

**位置**: `dataset/lm_dataset.py`

```python
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __getitem__(self, idx):
        conversations = self.data[idx]['conversations']

        # 使用chat_template格式化对话
        text = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )

        # 分词
        tokens = self.tokenizer.encode(text)
        tokens = tokens[:self.max_length]

        input_ids = torch.tensor(tokens)
        labels = input_ids.clone()

        # 关键：生成loss_mask，只对assistant部分计算loss
        loss_mask = self._create_loss_mask(tokens, conversations)
        labels[~loss_mask] = -100  # 非assistant部分不计算loss

        return input_ids, labels

    def _create_loss_mask(self, tokens, conversations):
        # 找到assistant回复的位置，只有这些位置计算loss
        mask = torch.zeros(len(tokens), dtype=torch.bool)
        # ... 根据特殊token标记assistant回复区域
        return mask
```

### Chat Template 格式

```python
# 原始对话
conversations = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
]

# 经过chat_template转换后
text = """<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮助你的？<|im_end|>"""
```

### Loss Mask 示意

```
tokens:    [BOS] user 你 好 [EOS] [BOS] assistant 你 好 ！ ... [EOS]
loss_mask: [ 0 ] [ 0] [0] [0] [ 0 ] [ 0 ]    [ 0 ]    [1] [1] [1] ... [ 1 ]

只有 loss_mask=1 的位置会计算loss
```

---

## 五、训练流程

SFT的训练流程和Pretrain几乎一样，核心差异在于：

### 5.1 加载预训练权重

```python
# init_model 函数
def init_model(lm_config, from_weight='pretrain', ...):
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    return model, tokenizer
```

### 5.2 使用SFTDataset

```python
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

### 5.3 训练循环（与Pretrain相同）

```python
for step, (input_ids, labels) in enumerate(loader):
    with autocast_ctx:
        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss

    scaler.scale(loss).backward()
    # ... 梯度累积、梯度裁剪、参数更新
```

---

## 六、运行示例

### 6.1 基本运行

```bash
python trainer/train_full_sft.py \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 1e-6 \
    --from_weight pretrain \
    --data_path ../dataset/sft_mini_512.jsonl
```

### 6.2 基于已有SFT继续训练

```bash
python trainer/train_full_sft.py \
    --from_weight full_sft \
    --epochs 1
```

### 6.3 断点续训

```bash
python trainer/train_full_sft.py \
    --from_resume 1
```

---

## 七、SFT数据准备

### 7.1 数据格式

```json
{"conversations": [{"role": "user", "content": "问题1"}, {"role": "assistant", "content": "回答1"}]}
{"conversations": [{"role": "user", "content": "问题2"}, {"role": "assistant", "content": "回答2"}, {"role": "user", "content": "追问"}, {"role": "assistant", "content": "追答"}]}
```

### 7.2 多轮对话

```json
{
  "conversations": [
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是人工智能的一个分支..."},
    {"role": "user", "content": "能举个例子吗？"},
    {"role": "assistant", "content": "比如垃圾邮件过滤..."}
  ]
}
```

### 7.3 带系统提示

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个有帮助的AI助手。"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！我是AI助手，有什么可以帮助你的？"}
  ]
}
```

---

## 八、注意事项

### 8.1 学习率很重要

```
太大 (>1e-5): 灾难性遗忘，模型"忘记"预训练学到的知识
太小 (<1e-7): 学不到对话能力
推荐: 1e-6 ~ 5e-6
```

### 8.2 数据质量 > 数据数量

SFT阶段：
- 高质量的1000条对话 > 低质量的10000条对话
- 确保对话格式正确
- 回答要准确、有帮助

### 8.3 过拟合风险

SFT数据通常较少，容易过拟合：
- 不要训练太多epochs
- 监控验证集loss
- 适当使用dropout

---

## 九、输出文件

```
out/
├── full_sft_512.pth          # SFT后的模型权重
└── checkpoints/
    └── full_sft_512_resume.pth  # 断点文件
```

---

## 十、验证效果

训练完成后，用 `eval_llm.py` 测试对话效果：

```bash
python eval_llm.py --weight full_sft
```

```
>>> 你好
你好！有什么可以帮助你的？

>>> 什么是机器学习？
机器学习是人工智能的一个分支，它使计算机能够从数据中学习...
```
