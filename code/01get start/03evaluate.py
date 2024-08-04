# evaluate实现对结果的评估
"""
1. 踩得坑:evaluate出现失灵的情况，解决措施：
    -首先进行源码安装：
        git clone https://github.com/huggingface/evaluate.git
        cd evaluate
        pip install -e .
    -需要切换到evaluate这个目录
2. 加载指标时，需要变成"./metrics/accuracy"
"""

## 1. 评估单个指标。直接用load指定
# 算法模型支持的指标和其对应的模型相关，具体指标需参考hug-face中的task模块

# import evaluate
# from evaluate import load
# accuracy = load("./metrics/accuracy")
# results = accuracy.compute(predictions=[0, 1, 1, 0], references=[0, 1, 0, 1])
# print(results)

## 2.评估多个指标。用combine

# import evaluate
# clf_metrics = evaluate.combine(["./metrics/accuracy","./metrics/f1"])
# results = clf_metrics.compute(predictions=[0, 1, 1, 0], references=[0, 1, 0, 1])
# print(results)

## 3. 对一个batch中多个结果进行判别

# import evaluate
# predictions=[0, 1, 1, 0]
# references=[0, 1, 0, 1]
# accuracy = evaluate.load("./metrics/accuracy")
# for pre, ref in zip(predictions, references):
#     accuracy.add(reference=ref, prediction=pre)
# results = accuracy.compute()
# print(results)

## 4. 对多个batch的结果进行判别

# import evaluate
# accuracy = evaluate.load("./metrics/accuracy")
# predictions=[[0, 1], [1, 0]]
# references=[[0, 1], [0, 1]]
# for pre, ref in zip(predictions, references):
#     accuracy.add_batch(predictions=pre, references=ref)
# result = accuracy.compute()
# print(result)

## 5. 展示雷达图
from evaluate.visualization import radar_plot
data = [
   {"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
   {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
   {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6}, 
   {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6}
   ]
model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]
plot = radar_plot(data=data, model_names=model_names)
print(plot)