# 答辩 Demo 讲稿

## 开场

“我们做的是跨域图文内容挪用检测。输入一张新闻图片和一段 caption，系统不仅判断是否错配，还输出错配类型和冲突字段。”

## 展示步骤

1. 打开 Gradio 页面；
2. 输入一条 Non-OOC 样例，展示事件字段一致；
3. 输入一条 location mismatch 样例，展示地点冲突；
4. 输入一条 temporal mismatch 或 event-type mismatch 样例；
5. 打开完整 JSON 输出，说明系统接口可接入真实 VDT/E3-VDT 模型。

## 强调

当前 demo 后端可替换；schema 已固定；复现实验和系统工程并行；最终接入严格 VDT/E3-VDT checkpoint。
