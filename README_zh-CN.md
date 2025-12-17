# PDF Craft OOMOL 功能块

中文 | [English](README.md)

轻松将 PDF 文档转换为现代化的可编辑格式。本 OOMOL 项目提供强大的 AI 驱动转换工具,可将 PDF 转换为 Markdown 或 EPUB 格式,非常适合文档管理、电子书制作和数字内容管理。

## 这个工具能做什么?

本项目提供两大核心功能:

### 📄 PDF 转 Markdown
将 PDF 文档转换为干净的 Markdown 文件,并自动提取所有图片。适用于:
- 创建技术文档
- 构建静态网站(Jekyll、Hugo、MkDocs)
- Git 版本控制工作流
- 内容管理系统
- 知识库和维基

### 📚 PDF 转 EPUB
将 PDF 转换为适合电子阅读器和移动设备的可重排 EPUB 电子书。理想用于:
- 数字图书出版
- 学术论文分发
- 技术手册转换
- 移动友好的阅读体验
- 电子阅读器图书馆管理

两个工具都使用先进的 GPU 加速 OCR 技术,可高精度处理数字 PDF 和扫描文档。

## 主要特性

- **智能文本识别**: GPU 驱动的 OCR 处理数字文本和扫描文档
- **图片提取**: 自动提取并组织所有图片,保持正确的链接关系
- **脚注保留**: 保持脚注的正确格式和引用关系
- **数学公式支持**: 将 LaTeX 公式转换为适当的显示格式
- **表格处理**: 保留复杂表格结构,提供多种渲染选项
- **灵活的 OCR 模型**: 从 5 种模型大小中选择,平衡速度和精度
- **GPU 优化**: 可调节的内存使用和精度设置,实现最佳性能
- **分析可视化**: 可选的诊断图表用于质量保证
- **元数据自定义**: 为 EPUB 输出设置书名、作者和渲染偏好

## 可用功能块

### PDF 转 Markdown 功能块

将 PDF 文档转换为 Markdown 格式,自动提取和组织图片。

**需要提供的内容:**
- **PDF 文件**: 要转换的 PDF 文档
- **输出位置**(可选): 保存 Markdown 文件和图片的位置(默认为会话目录)

**可选设置:**
- **包含脚注**: 在输出中保留或移除脚注(默认:不包含)
- **OCR 模型大小**: 选择精度与速度的平衡(选项: tiny、small、gundam、base、large;默认: base)
- **生成分析**: 创建显示转换质量和性能的诊断图表
- **优化级别**: 选择 GPU 精度策略(balanced 或 quality;默认: balanced)
- **GPU 内存**: 控制使用多少 GPU 内存(10-100%;默认: 90%)

**输出结果:**
- 一个 ZIP 压缩包,包含:
  - 清晰的 Markdown 文件,包含格式化的文本、标题、列表、表格和公式
  - 图片文件夹,包含所有提取的原始质量图形
  - Markdown 和图片之间正确链接的引用

**最适合:** 文档系统、静态网站、基于 Git 的工作流、内容编辑

---

### PDF 转 EPUB 功能块

将 PDF 文档转换为适合电子阅读器和移动设备的 EPUB 电子书格式。

**需要提供的内容:**
- **PDF 文件**: 要转换的 PDF 文档
- **输出位置**(可选): 保存 EPUB 文件的位置(默认为会话目录)

**可选设置:**
- **书名**: 设置在电子阅读器中显示的书名元数据
- **作者**: 指定作者姓名(多个作者用逗号分隔)
- **包含脚注**: 保留或移除脚注(默认:不包含)
- **OCR 模型大小**: 选择处理质量(tiny、small、gundam、base、large;默认: base)
- **表格渲染**: 如何显示表格(HTML 或 Markdown;默认: HTML)
- **LaTeX 渲染**: 如何显示数学公式(MathML 或 LaTeX;默认: MathML)
- **生成分析**: 创建转换质量可视化图表

**输出结果:**
- 一个完全格式化的 EPUB 文件,可在任何电子阅读器应用中打开
- 正确结构化的章节和导航
- 用于图书馆组织的嵌入元数据
- 适应屏幕尺寸的可重排文本

**最适合:** 电子书、学术论文、数字出版、移动阅读

## 快速开始

### 前置要求

使用这些功能块,您需要:
- 支持 CUDA 的 NVIDIA GPU(OCR 处理必需)
- 足够的 GPU 内存(建议至少 6-8GB 显存)
- 模型缓存的磁盘空间(约 3-5GB)

### 安装

项目在 OOMOL 中首次加载时会自动处理安装:

1. **系统依赖**: 安装用于 PDF 处理的 poppler-utils
2. **Python 环境**: 设置支持 CUDA 12.x 的 PyTorch
3. **AI 模型**: 首次使用时下载并缓存 OCR 模型
4. **Python 库**: 通过 poetry 安装所有必需的包

无需手动设置 - 只需加载项目即可开始转换!

### 使用功能块

#### 在 OOMOL 工作流中使用

在工作流中引用这些功能块:
- `self::pdf-to-markdown` 用于 Markdown 转换
- `self::pdf-to-epub` 用于 EPUB 转换

**示例工作流:**

```yaml
nodes:
  - node_id: convert-to-markdown#1
    task: self::pdf-to-markdown
    inputs_from:
      - handle: pdf_path
        value: "/path/to/your/document.pdf"
      - handle: output_path
        value: "/oomol-driver/oomol-storage/output/document.md"
      - handle: includes_footnotes
        value: true
      - handle: ocr_size
        value: "base"

  - node_id: convert-to-epub#1
    task: self::pdf-to-epub
    inputs_from:
      - handle: pdf_path
        value: "/path/to/your/document.pdf"
      - handle: output_path
        value: "/oomol-driver/oomol-storage/output/book.epub"
      - handle: book_title
        value: "我的技术书籍"
      - handle: book_authors
        value: "张三, 李四"
      - handle: table_render
        value: "HTML"
      - handle: latex_render
        value: "MathML"
```

#### 测试工作流

示例测试工作流位于 [flows/test-pdf-conversion/flow.oo.yaml](flows/test-pdf-conversion/flow.oo.yaml)。更新 `pdf_path` 值以指向您自己的 PDF 文件。

## 选择合适的 OCR 模型

不同的 OCR 模型在处理速度和精度之间提供权衡:

| 模型大小 | 速度 | 质量 | GPU 内存 | 最适合 |
|---------|------|------|---------|--------|
| **tiny** | 最快 | 最低 | ~2GB | 高质量 PDF,简单布局 |
| **small** | 快 | 良好 | ~4GB | 标准文档,良好扫描质量 |
| **gundam** | 平衡 | 很好 | ~6GB | 通用目的,推荐给大多数用户 |
| **base** | 中等 | 高 | ~8GB | 默认选项,出色的精度(推荐) |
| **large** | 最慢 | 最高 | ~12GB | 复杂布局,差扫描,最高精度 |

**建议**: 从 **base**(默认)开始以获得最佳结果。使用 **gundam** 以获得更快的处理速度和良好的质量,或使用 **large** 处理具有挑战性的文档。

## GPU 优化指南

### 优化级别

- **Balanced**(平衡,默认): 在现代 GPU(RTX 30/40 系列)上使用 bfloat16 精度以获得最佳速度/质量平衡
- **Quality**(质量): 使用 float16 精度以获得稍高的精度,可能较慢

### GPU 内存管理

`gpu_memory_fraction` 设置控制分配多少 GPU 内存:

- **0.9(90%,默认)**: 专用处理的最大性能
- **0.7(70%)**: 良好平衡,为其他应用程序留出内存
- **0.5(50%)**: 保守设置,适合共享 GPU 环境

**提示**: 如果遇到内存不足错误,请减小此值或使用较小的 OCR 模型。

## 存储位置

- **模型缓存**: `/oomol-driver/oomol-storage/pdf-craft-models-cache`(自动管理)
- **输出文件**: 建议对运行时文件使用 `/oomol-driver/oomol-storage/` 下的路径
- **会话文件**: 未指定输出路径时,默认输出到特定会话目录

## 技术细节

### 系统要求

- Python 3.10-3.12
- 支持 CUDA 12.x 的 NVIDIA GPU
- PyTorch 2.9.0 与 CUDA 加速
- 用于 PDF 处理的 Poppler 工具

### 主要依赖

- [pdf-craft](https://github.com/oomol-lab/pdf-craft): 核心 PDF 转换库
- PyTorch: 用于 OCR 模型的深度学习框架
- Transformers: Hugging Face 模型支持
- PyMuPDF: PDF 解析和操作
- EbookLib: EPUB 生成

### 架构

两个功能块都使用 GPU 加速管道:
1. **PDF 解析**: 提取页面和基本结构
2. **OCR 处理**: 使用可配置模型进行 AI 驱动的文本识别
3. **图片提取**: 自动图片检测和导出
4. **内容组装**: 以目标格式重构文档结构
5. **格式生成**: 创建最终的 Markdown 或 EPUB 输出

### 性能特征

- **处理速度**: 每页约 1-3 秒(因 OCR 模型和 GPU 而异)
- **GPU 利用率**: OCR 处理期间通常为 70-95%
- **内存使用**: 根据模型大小使用 4-12GB GPU 内存
- **精度**: 大多数文档的文本识别精度为 95-99%

## 故障排除

### 常见问题

**内存不足错误**
- 解决方案: 将 `gpu_memory_fraction` 减小到 0.7 或 0.5
- 替代方案: 使用较小的 OCR 模型(small 或 gundam)

**低质量输出**
- 解决方案: 切换到更大的 OCR 模型(base 或 large)
- 替代方案: 启用分析图表以诊断质量问题

**处理缓慢**
- 解决方案: 使用较小的 OCR 模型(gundam 或 small)
- 替代方案: 确保 GPU 优化级别设置为 "balanced"

**缺少脚注**
- 解决方案: 将 `includes_footnotes` 设置为 `true`

**表格格式不正确(EPUB)**
- 解决方案: 尝试在 HTML 和 Markdown 渲染模式之间切换

## 贡献

本项目基于 [pdf-craft](https://github.com/oomol-lab/pdf-craft) 构建。对于与核心转换引擎相关的问题或贡献,请访问 pdf-craft 仓库。

## 许可证

请参阅 [pdf-craft 许可证](https://github.com/oomol-lab/pdf-craft) 了解使用条款和条件。

---

**需要帮助?** 查看 [OOMOL 文档](https://github.com/oomol-flows/pdf-craft-next) 或在 GitHub 上提交问题。
