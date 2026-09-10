# 发布说明

## 推荐结构

一个 GitHub 仓库，main 保留本综合版本；原基础版、OCR 版等作为历史提交并用标签标注，需要下载入口时再发布 Releases。
仅语义与混合检索继续作为同一应用的选项，不复制两个项目目录。
本包只是可发布源码，尚未推送仓库，也未创建标签或 Releases。

## 已跑通 Windows 环境的版本快照

在 D:\deepseek-rag 中复制本包后，用 CMD 执行：

```bat
.venv\Scripts\activate.bat
python -m pip check
python scripts\export_environment.py
```

脚本只导出包名和版本，不导出密钥或下载地址，生成 requirements-windows-lock.txt。
这只是已安装包快照，仍需新虚拟环境安装验收才可声明可复现；尤其带 +cpu 的 torch/torchvision 版本需要官方 CPU 源。
不要使用交付机器的包列表冒充用户电脑环境。本包不附带未经验证的 Windows 锁定文件。

## 截图

从实际运行的应用截图：知识问答及来源、文档管理、低相似度提醒。遮盖 API Key、私人文件名和私人原文。
可保存在 docs/images/，再在 README 用相对路径引用。当前包没有伪造实际问答截图。

## 创建仓库并推送

在 GitHub 账号下创建空仓库 deepseek-rag，选择需要的可见性，不自动生成 README。
回到项目目录执行 git init，配置你的提交身份，添加发布源码并检查 git status。
优先明确添加源码和文档，切勿提交 .env、data、.venv、模型缓存或私有资料。
仓库地址使用 GitHub 实际创建后给出的地址。认证时不把令牌写入命令行或远程地址。

本次检查已连接账号为 ddding121，但接口未返回可访问仓库；需提供目标仓库链接后继续上传。

## 忽略规则

.gitignore 排除密钥、数据库、虚拟环境、缓存和文档。docs/images 下允许公开截图。
忽略规则不移除已经进入 Git 历史的内容；如果曾提交密钥，应先撤销密钥再清理历史。
