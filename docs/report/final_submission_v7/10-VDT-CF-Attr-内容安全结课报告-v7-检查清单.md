# v7 实验报告检查清单

- DOCX：`docs/report/final_submission_v7/10-VDT-CF-Attr-内容安全结课报告-v7.docx`
- PDF：`docs/report/final_submission_v7/10-VDT-CF-Attr-内容安全结课报告-v7.pdf`
- PDF 渲染检查：已渲染 `render_v7/page-1.png` 至 `page-9.png`，封面、目录、图表页、结果页、结尾页已检查。
- 说明：documents 技能自带 LibreOffice 渲染器在本机缺少 `soffice`，最终采用 Word COM 导出 PDF + Poppler 渲染进行视觉 QA。

- [x] DOCX 文件存在：10-VDT-CF-Attr-内容安全结课报告-v7.docx / 260074 bytes
- [x] PDF 文件存在：10-VDT-CF-Attr-内容安全结课报告-v7.pdf / 533597 bytes
- [x] Word 可打开与目录：Word COM 只读：Pages=9，TablesOfContents=1，Tables=8，InlineShapes=4，Comments=0，Revisions=0
- [x] 封面任课老师：任课老师：任延珍
- [x] 封面无表格：全文仅 8 张正文数据表，首页未新增封面表格
- [x] 目录域存在：XML 中存在 TOC 字段
- [x] 表格数量：8 张
- [x] 图片嵌入：media_count=5
- [x] 图题数量：3 个：图1  VDT-CF-Attr 系统总体框架；图2  单源多变体与分组切分流程；图3  no-true-context 特征提取与归因头
- [x] 表题数量：8 个：表1  系统门控规则；表2  可控反事实编辑与标签映射；表3  分组切分泄漏审计；表4  VDT strict 主分类复现结果；表5  no-true-context 训练规模实验；表6  different-event 分布修正实验；表7  真实 OOC 人工类型分布；表8  真实 OOC no-true-context 评测
- [x] 参考文献编号：正文/列表编号=[1, 2, 3, 4, 5, 6, 7, 8, 9], 列表=9 条
- [x] rsid 清理：package_has_rsid=False
- [x] 残留占位/本机路径/模板占位：命中：无
- [x] AI报告腔高频词：命中：无
- [x] 表格竖线/内部竖线：vertical_border_count=0
- [x] 表格底纹：shading_count=0
- [x] 表格单元格大段文字：无
- [x] PDF 页数：pdfinfo Pages=9
- [x] PDF 页面尺寸：A4
- [x] PDF Suspects：Suspects: no
- [x] PDF 渲染页数：9 张 page-*.png

## 本轮关键修订

- 封面填写 `任课老师：任延珍`，四行信息改为同一缩进与制表位，首页不使用表格。
- `1.4 参考文献` 改为编号列表，并在技术现状、VDT 主分类、no-true-context 特征、实验环境和演示入口处加入引用编号。
- 表 3 移至第 6 页页首完整展示，不再跨页拆分。
- 第 5 章改为 `讨论与改进方向`，第 6.2 节压缩为工作链路、关键数值、限制和下一步。

## pdfinfo 摘要
```
Suspects:        no
Pages:           9
Encrypted:       no
Page size:       595.32 x 841.92 pts (A4)
```
