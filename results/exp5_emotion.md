# 实验 5b：预处理是否抹掉语音情绪

模型 `iic/emotion2vec_plus_large`；指标 = emotion2vec P(angry) 均值（辩论片段以 angry 为主）。

clean 片段 26 条，其中 angry=17。

## A 降噪伪影（clean → 降噪，无噪声混淆）

| 处理 | angry 片段 P(angry) 均值 | angry→非angry 翻转数 |
|---|---|---|
| clean（基线） | 0.893 | - |
| frcrn | 0.848 | 1/17 |
| specsub | 0.652 | 4/17 |

## A' 真实管线：噪声 + 降噪后 P(angry)（angry 片段均值）

| 方法 | babble_15dB | babble_5dB | babble_0dB | white_15dB | white_5dB | white_0dB |
|---|---|---|---|---|---|---|
| frcrn | 0.802 | 0.221 | 0.028 | 0.885 | 0.810 | 0.484 |
| specsub | 0.697 | 0.469 | 0.265 | 0.650 | 0.356 | 0.208 |

## B 分离（clean 噪声条件）：源 vs 分离两路 P(angry) 均值

源说话人均值：0.628

| 链路 | no | light | heavy |
|---|---|---|---|
| L3_sep | 0.369 | 0.468 | 0.349 |
| L4_sep | 0.148 | 0.096 | 0.101 |

图：`results/exp5_emotion_drift.png`
