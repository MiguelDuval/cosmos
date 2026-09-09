import base64
import json
import os
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


DEFAULT_ENDPOINT = "https://ai.api.nvidia.com/v1/genai/nvidia/cosmos3-nano"


class NvidiaCosmos3Nano:
    """Generate a short video through NVIDIA's hosted Cosmos 3 Nano API."""

    CATEGORY = "NVIDIA/Cosmos"
    FUNCTION = "generate"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A cinematic shot of a small robot walking through a futuristic city at night.",
                    },
                ),
                "resolution": (["256", "480", "720"], {"default": "480"}),
                "aspect_ratio": (
                    ["16:9", "4:3", "1:1", "3:4", "9:16"],
                    {"default": "16:9"},
                ),
                "num_output_frames": (
                    "INT",
                    {"default": 25, "min": 5, "max": 197, "step": 4},
                ),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 100}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 6.0, "min": 1.0, "max": 7.0, "step": 0.1},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": "blurry, distorted, low quality"},
                ),
                "image": ("IMAGE",),
            },
        }

    @staticmethod
    def _api_key():
        key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")
        if not key:
            raise RuntimeError(
                "NVIDIA API key is missing. Set NVIDIA_API_KEY in the environment, "
                "then restart ComfyUI. Do not put the key in the workflow or GitHub repository."
            )
        return key.strip()

    @staticmethod
    def _endpoint():
        return os.environ.get("COSMOS3_INFER_URL", DEFAULT_ENDPOINT).strip().rstrip("/")

    @staticmethod
    def _encode_image(image):
        # ComfyUI IMAGE is [B,H,W,C], float32 in 0..1.
        frame = image[0].detach().cpu().numpy()
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(frame[:, :, :3], mode="RGB")
        from io import BytesIO

        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _post_json(url, payload, api_key, timeout=1800):
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "ComfyUI-NVIDIA-Cosmos/1.0",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:4000]
            if exc.code == 401:
                raise RuntimeError("NVIDIA API returned 401 Unauthorized. Check NVIDIA_API_KEY.") from exc
            if exc.code == 403:
                raise RuntimeError(
                    "NVIDIA API returned 403 Forbidden. Check that the NVIDIA API key has access "
                    "to the hosted Cosmos endpoint."
                ) from exc
            if exc.code == 429:
                raise RuntimeError(
                    "NVIDIA API returned 429 Too Many Requests. The free hosted endpoint is rate limited; "
                    "wait and try again."
                ) from exc
            raise RuntimeError(f"NVIDIA API returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach NVIDIA API: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("NVIDIA API request timed out.") from exc

        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("NVIDIA API returned a non-JSON response.") from exc

    @staticmethod
    def _decode_video(response):
        value = response.get("b64_video")
        if not value:
            # Be defensive about a few common wrapper shapes.
            for container in (response.get("data"), response.get("output")):
                if isinstance(container, dict) and container.get("b64_video"):
                    value = container["b64_video"]
                    break
        if not value or not isinstance(value, str):
            keys = ", ".join(sorted(response.keys()))
            raise RuntimeError(
                "NVIDIA API response did not contain 'b64_video'. "
                f"Response keys: {keys or '<none>'}"
            )
        try:
            return base64.b64decode(value, validate=True)
        except Exception as exc:
            raise RuntimeError("NVIDIA API returned invalid base64 video data.") from exc

    @staticmethod
    def _save_video(video_bytes):
        import folder_paths

        output_dir = Path(folder_paths.get_output_directory())
        target_dir = output_dir / "nvidia_cosmos"
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"cosmos3_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
        path = target_dir / filename
        path.write_bytes(video_bytes)
        return path

    def generate(
        self,
        prompt,
        resolution,
        aspect_ratio,
        num_output_frames,
        fps,
        steps,
        guidance_scale,
        seed,
        negative_prompt="",
        image=None,
    ):
        if num_output_frames < 5 or (num_output_frames - 1) % 4 != 0:
            raise ValueError(
                "num_output_frames must follow Cosmos 3's 4k+1 cadence: 5, 9, 13, 17, 21, 25, ..."
            )

        caps = {"256": 397, "480": 297, "720": 197}
        if num_output_frames > caps[resolution]:
            raise ValueError(
                f"At {resolution}p, num_output_frames cannot exceed {caps[resolution]} for Cosmos 3."
            )

        payload = {
            "prompt": prompt,
            "seed": int(seed),
            "guidance_scale": float(guidance_scale),
            "steps": int(steps),
            "resolution": f"{resolution}_{aspect_ratio.replace(':', '_')}",
            "num_output_frames": int(num_output_frames),
            "fps": float(fps),
        }
        if negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt
        if image is not None:
            payload["image"] = self._encode_image(image)

        api_key = self._api_key()
        endpoint = self._endpoint()
        print(f"[NVIDIA Cosmos] POST {endpoint} ({'I2V' if image is not None else 'T2V'})")
        response = self._post_json(endpoint, payload, api_key)
        video_bytes = self._decode_video(response)
        path = self._save_video(video_bytes)

        # Import ComfyUI's native VIDEO wrapper lazily so repository tooling can
        # inspect this package without needing the ComfyUI runtime installed.
        try:
            from comfy_api.latest import InputImpl
            video = InputImpl.VideoFromFile(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"Video was generated successfully at {path}, but this ComfyUI build "
                "does not expose the native VIDEO API (comfy_api.latest.InputImpl.VideoFromFile). "
                "Update ComfyUI to a current version."
            ) from exc

        return (video,)


NODE_CLASS_MAPPINGS = {
    "NvidiaCosmos3Nano": NvidiaCosmos3Nano,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NvidiaCosmos3Nano": "NVIDIA Cosmos 3 Nano (Hosted API)",
}
