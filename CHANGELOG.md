# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-04-02

### Added
- FastAPI proxy with two endpoints: `/v1/transcribe` (generic) and `/api/willow` (Willow-compatible)
- OpenVINO-accelerated whisper.cpp integration (2.5x faster encoder on Intel CPUs)
- Automated install script (builds whisper.cpp, downloads models, generates configs)
- systemd service files for both whisper-server and the proxy
- Health check endpoint with whisper-server status
- Willow/WIS integration guide with nginx snippet
- Configurable via `.env` file
