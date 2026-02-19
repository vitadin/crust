package main

import "embed"

//go:embed services/lfm25_server/config.yaml
//go:embed services/lfm25_server/pyproject.toml
//go:embed services/lfm25_server/uv.lock
//go:embed services/lfm25_server/main.py
//go:embed services/lfm25_server/download_model.py
//go:embed all:services/lfm25_server/lfm25_server
var ServerAssets embed.FS
