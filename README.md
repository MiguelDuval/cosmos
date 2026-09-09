# ComfyUI NVIDIA Cosmos 3 Nano

A lightweight ComfyUI custom node that sends Text-to-Video or Image-to-Video requests to NVIDIA's hosted Cosmos 3 Nano endpoint and returns the generated MP4 as ComfyUI's native `VIDEO` type.

## Important

This node uses NVIDIA's **hosted** API. Your PC does not run Cosmos locally and the GTX 1650 is not used for model inference.

The current default endpoint is:

`https://ai.api.nvidia.com/v1/genai/nvidia/cosmos3-nano`

NVIDIA's Build page labels Cosmos3-Nano as a downloadable/free endpoint and notes that free API requests may be rate limited or throttled. The endpoint can be overridden with `COSMOS3_INFER_URL` if NVIDIA changes the hosted route or you want to use a self-hosted Generator NIM.

## Install in ComfyUI Portable (Windows)

Copy/clone this repository into:

`ComfyUI_windows_portable\\ComfyUI\\custom_nodes\\cosmos`

Then restart ComfyUI.

No extra pip package is required by this node beyond the normal ComfyUI environment.

## NVIDIA API key

Create an API key on NVIDIA Build and set it as a Windows environment variable. Do **not** put the key into this repository or a ComfyUI workflow.

PowerShell for the current terminal session:

```powershell
$env:NVIDIA_API_KEY="nvapi-PASTE_YOUR_KEY_HERE"
```

To persist it for future terminals:

```powershell
setx NVIDIA_API_KEY "nvapi-PASTE_YOUR_KEY_HERE"
```

After `setx`, start a new terminal and launch ComfyUI from that terminal, or restart ComfyUI if you launch it another way.

The node also accepts `NGC_API_KEY` as a fallback environment variable.

## Node

Search the node menu for:

**NVIDIA Cosmos 3 Nano (Hosted API)**

Inputs:

- Prompt
- Resolution: 256 / 480 / 720
- Aspect ratio: 16:9 / 4:3 / 1:1 / 3:4 / 9:16
- Output frames (Cosmos 4k+1 cadence)
- FPS
- Steps
- Guidance scale
- Seed
- Optional negative prompt
- Optional input image for Image-to-Video

The output is a native ComfyUI `VIDEO`, so it can be connected directly to ComfyUI's built-in **Save Video** node.

## Endpoint override

If NVIDIA changes the hosted endpoint, set:

```powershell
$env:COSMOS3_INFER_URL="https://your-endpoint/v1/infer"
```

For a local/self-hosted Cosmos 3 Generator NIM, the documented API is `POST /v1/infer` and returns JSON containing `b64_video`.

## Troubleshooting

### 401
The API key is missing, invalid, or not available to the ComfyUI process. Check `NVIDIA_API_KEY` and restart ComfyUI.

### 403
The key/account does not have access to the hosted endpoint.

### 429
The hosted free endpoint is rate limited. Wait and retry.

### 404
Check `COSMOS3_INFER_URL`. NVIDIA's Build hosted catalog route and the local Generator NIM route are different concepts; do not blindly replace the hosted URL with `http://127.0.0.1:8000/v1/infer` unless you are actually running a local NIM.

### Video generated but node cannot return VIDEO
Update ComfyUI to a current version that exposes the native `VIDEO` API (`comfy_api.latest.InputImpl.VideoFromFile`).
