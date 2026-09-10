# DeepSeek RAG 垂直知识库问答

上传 Word、PDF 或图片，建立可保存的本地知识库，通过 DeepSeek API 根据检索原文回答，并显示来源。

当前交付：**V3 原页面布局 + 低相似度提醒**。保留“知识问答 / 文档管理 / 备份与恢复”三个页签。

## 功能

- 文字型 PDF、DOCX、PNG/JPG 等常见图片导入；扫描 PDF、Word 内嵌位图本地 OCR。
- BGE 中文向量与 BM25 关键词混合检索，可切换仅语义模式。
- 可选 BGE 模型重排序，默认关闭，首次启用需下载模型。
- 按句子、段落和标题切块；回答引用文件名、PDF 页码或 Word 段落范围。
- SQLite 保存原文、文字片段和向量；多个知识库、增量更新、文档替换/删除及备份。
- 当全库最高语义相似度低于阈值时，提示核对原文后适当降低阈值；不自动修改参数。

## Windows CMD 快速启动

需要 Python 3.11。解压或克隆后，在项目目录执行：

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m streamlit run app.py --server.maxUploadSize 50
```

浏览器打开 http://localhost:8501，在左侧输入自己的 DeepSeek API Key，新建知识库，在文档管理中上传并保存资料，再提问。
首次出现 Streamlit 邮箱提示时可直接按回车跳过。保持 CMD 窗口打开。

也可以复制 `.env.example` 为 `.env` 填入配置；不要提交 `.env`。已有虚拟环境可双击 `start.bat` 启动。

## 升级与资料保存

从 V3 或后续诊断版切换至此版，停止程序后覆盖本包源码，保留 `.venv`、`.env`、`data`。
已有 V3 索引无需重建；V2 旧切块可在文档管理中使用已保存原文件重建。
原文件和索引保存在 `data/knowledge.db`；删除整个项目会删除数据。请通过备份页下载备份到项目目录之外。

## 检索提醒示例

> 本次最高语义相似度为 0.380，低于设定阈值 0.50。可以先核对相关原文，再适当降低左侧“语义候选最低相似度”，然后重新发送问题。

这是提示文案示例，不是实际模型测试结果。相似度并非答案可信度；混合检索可通过关键词补充低语义分片段。

## 测试与可复现性

```bat
python -m unittest discover -s tests -v
```

本次交付环境复跑 27 项测试通过（Linux / Python 3.12）；Windows / Python 3.11 的 CI 尚未远程运行。

自动化测试使用固定向量、模拟重排序和临时数据库，覆盖检索逻辑、解析、原文持久化、增量替换、回滚、备份、旧索引重建、提醒及页面操作。
真实 DeepSeek 调用、重排序模型推理效果、用户 Windows 环境和真实中文文档准确率仍需实测。CI 配置已提供，远程运行结果以 GitHub Actions 为准。

`requirements.txt` 是运行依赖范围，尚不是用户 Windows 环境锁定文件。
`requirements-test.txt` 仅固定交付环境实际使用的测试包版本。导出已跑通的 Windows 依赖方法见 [发布说明](docs/RELEASE.md)。

## 限制

- 每库最多 10 个文件，原文件合计 50 MB、10000 片段；每个 PDF 最多 400 页。
- OCR 读取文字，不理解照片场景、图表关系或复杂公式；自动模式可能漏掉混排图片文字，可选择全页 OCR。
- 本地单用户应用，没有在线登录/多租户权限体系。默认仅监听 127.0.0.1。
- 聊天记录不跨会话保存，没有连续追问改写、PDF 页内高亮或 OCR 断点续传。
- 提问和命中的原文片段发送给 DeepSeek；OCR 在本地执行。请确认资料允许发送到该服务。

## 项目文件

| 文件 | 用途 |
| --- | --- |
| app.py | V3 页面与低相似度提醒 |
| rag.py / chunking.py | 解析、切块和回答 |
| retrieval.py | 混合检索、重排序及提醒条件 |
| ocr_engine.py | 本地 OCR |
| storage.py / ingestion.py | 保存、备份和增量导入 |
| tests/ | 自动化验证 |

## 更多说明

- [使用与排错](docs/USAGE.md)
- [真实资料验收](docs/ACCEPTANCE.md)
- [发布与版本管理](docs/RELEASE.md)
- [本次发布说明](CHANGELOG.md)

## 官方参考

[DeepSeek API](https://api-docs.deepseek.com/) · [BGE 中文模型](https://huggingface.co/BAAI/bge-small-zh-v1.5) · [BGE 重排序](https://huggingface.co/BAAI/bge-reranker-base) · [RapidOCR](https://github.com/RapidAI/RapidOCR)

发布许可证尚未选择；依赖和模型分别遵循其各自许可证。
