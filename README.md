# AI Learning Notebook

大模型架构、核心模块、训练系统与性能工程个人知识库。

## Markdown 文档站

知识库以 `docs/` 下的 Markdown 为唯一内容源，由 Zensical 构建。全站按“模型架构、核心模块、训练系统、性能工程”组织；并行训练作为训练系统下的一本专题书维护。

- 首页与领域入口：`docs/index.md`、`docs/{models,components,training,engineering}/`
- 并行训练正文：`docs/training/parallelism/*.md`
- 并行训练图片：`docs/training/parallelism/assets/`
- 全书导航：`zensical.toml`

本地预览：

```bash
python -m venv .venv-docs
.venv-docs/bin/pip install -r requirements-docs.txt
.venv-docs/bin/zensical serve
```

浏览器访问 `http://127.0.0.1:8000`。保存 Markdown 后页面会自动刷新。

检查生产构建：

```bash
.venv-docs/bin/zensical build --clean --strict
```

生成的 `site/` 是构建产物，不需要手工修改或提交。

## GitHub Pages

`main` 分支的每次推送都会触发 `.github/workflows/pages.yml`：严格构建知识库、上传 `site/`，然后发布到 GitHub Pages。

线上地址：<https://husichao666.github.io/AI_learning_notebook/>

首次启用时，在仓库的 **Settings → Pages → Build and deployment → Source** 中选择 **GitHub Actions**。之后只需维护并提交 `docs/` 下的 Markdown。
