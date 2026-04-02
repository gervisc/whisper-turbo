# Willow / WIS Integration

This guide shows how to route Willow Inference Server's speech-to-text through whisper-turbo for faster transcription, while keeping WIS for TTS and everything else.

## How It Works

```
ESP32 --HTTPS--> WIS nginx :19000
                   |
                   +-- POST /api/willow --> whisper-turbo :19004 --> whisper.cpp :19003
                   |
                   +-- /api/tts ---------> Coqui TTS (unchanged)
                   +-- /api/* -----------> WIS (unchanged)
                   +-- /* ---------------> static files (unchanged)
```

The ESP32 still talks to port 19000. Only STT is rerouted. TTS, docs, and everything else goes to WIS as before.

## Setup

### Prerequisites

- whisper-turbo installed and running (see main [README](../README.md))
- Willow Inference Server running in Docker
- WIS nginx listening on port 19000

### 1. Find your WIS nginx.conf

Typically at:
```
~/willow-inference-server/nginx/nginx.conf
```

### 2. Replace the /api/willow location block

Find this block in nginx.conf:
```nginx
location /api/willow {
    proxy_buffering off;
    proxy_request_buffering off;
    proxy_pass http://keepalive-wis/api/willow;
}
```

Replace it with the contents of [nginx-snippet.conf](nginx-snippet.conf):
```nginx
location /api/willow {
    proxy_buffering off;
    proxy_request_buffering off;
    proxy_pass http://172.17.0.1:19004/api/willow;
}
```

> **Note:** `172.17.0.1` is Docker's default bridge gateway IP. This is how the nginx container (inside Docker) reaches whisper-turbo (on the host). If your Docker bridge uses a different IP, check with: `docker network inspect bridge | grep Gateway`

### 3. Ensure whisper-turbo binds to the right address

If WIS runs in Docker, the proxy must be reachable from inside the container. In your `.env`, set:
```
PROXY_HOST=0.0.0.0
```

This binds to all interfaces so the Docker container can reach it. The proxy only processes audio — no sensitive data is exposed.

### 4. Restart WIS nginx

```bash
docker restart <your-wis-nginx-container>
```

### 5. Test

```bash
# Send a test WAV through the full chain (HTTPS through nginx to proxy)
curl -sk -X POST https://localhost:19000/api/willow \
    -H "x-audio-sample-rate: 16000" \
    -H "x-audio-bits: 16" \
    -H "x-audio-channel: 1" \
    -H "x-audio-codec: wav" \
    --data-binary @test.wav
```

Expected response:
```json
{"language": "en", "text": "your transcribed text", "infer_time": 500, "infer_speedup": 6.0, "audio_duration": 3000}
```

## Reverting

To go back to WIS's built-in STT, change the nginx location block back to:
```nginx
location /api/willow {
    proxy_buffering off;
    proxy_request_buffering off;
    proxy_pass http://keepalive-wis/api/willow;
}
```

Then restart the WIS nginx container.
