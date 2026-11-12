"""Stream actual MuJoCo frames to a browser-compatible MP4 using FFmpeg."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import mujoco as mj


class VideoRecorder:
    def __init__(self, model, path, *, fps=30, width=1280, height=800, camera=None, options=None):
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("Recording requires ffmpeg on PATH")
        self.path, self.fps, self.frames = Path(path), fps, 0
        self.camera, self.options = camera, options
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mj.Renderer(model, height=height, width=width)
        self.process = subprocess.Popen([
            executable, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(self.path),
        ], stdin=subprocess.PIPE)

    def write(self, data, decorate):
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.options)
        decorate(self.renderer.scene)
        self.process.stdin.write(self.renderer.render().tobytes())
        self.frames += 1

    def close(self):
        try:
            self.process.stdin.close()
            code = self.process.wait()
            if code:
                raise RuntimeError(f"FFmpeg exited with status {code}")
        finally:
            self.renderer.close()

    def report(self):
        return {"file": self.path.name, "fps": self.fps, "frames": self.frames,
                "duration_seconds": self.frames/self.fps, "mujoco_version": mj.__version__}
