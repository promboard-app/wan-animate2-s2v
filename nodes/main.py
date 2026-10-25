"""promboard.wan-animate2-s2v — Wan Animate-2(동작 옮기기) · Wan2.2 S2V(말하는 영상) 노드 코드.
🔴 검증 안 됨 · 서버급: 이 코드는 실제로 돌려 보지 않았다(필요한 그래픽카드 메모리가 이 PC 보다 훨씬 크다).

노드 하나(wan-a2s)가 모드에 따라 두 모델 중 하나를 쓴다.
  animate2 · animate2-fast  Wan-AI/Wan2.2-Animate-2-14B(-Distilled)-Diffusers. 캐릭터 그림 + 움직임 영상 → 캐릭터가 영상 속 사람처럼
                            움직이는 영상. Animate-2 는 관절 전처리 없이 움직임 영상을 그대로 받는다.
                            diffusers 0.42 Modular Diffusers 블록(WanAnimate2Blocks · WanAnimate2DistilledBlocks)을 저장소 파일로
                            직접 채워 쓴다 — 저장소의 modular_model_index.json 은 허깅페이스 주소(refs/pr/2)를 가리켜 오프라인으로 못 읽는다.
                            블록 기본값: 일반 40단계 · 증류 10단계, 24fps, 81프레임 구간, 넓이 800×640(비율은 그림을 따른다).
  s2v                       Wan-AI/Wan2.2-S2V-14B. 캐릭터 그림 + 목소리(소리 파일) + 프롬프트 → 그 소리에 맞춰 말하는 영상.
                            diffusers 파이프라인이 없어 Wan2.2 공식 코드(wan/speech2video.py 와 그 모듈, Apache-2.0)를 nodes/wan/ 에 옮겨 쓴다.
                            공식 generate.py s2v 기본값: 클립당 80프레임, 40단계, guidance 4.5, shift 3, unipc, 16fps,
                            길이는 소리를 따른다(num_clip 없음 = 소리 끝까지). 결과 영상에 입력 소리를 넣는다(공식 merge_video_audio 와 같은 일).
                            공식 코드에서 바꾼 것: flash-attn 이 없을 때 PyTorch SDPA 로 같은 계산(modules/attention.py),
                            decord 를 쓰는 곳(이어 만들기 · 포즈 영상)만 그때 불러온다(speech2video.py). 둘 다 'PromBoard:' 주석.
진행률 · 취소: 두 모델 모두 단계 콜백이 없다. Animate-2 는 디노이즈 반복이 구간마다 여는 tqdm 을(_SegmentProgress),
  S2V 는 변환기 forward 를(_s2v) 이 실행 동안만 바꿔 끼워 단계마다 취소를 보고 진행률을 알린다.

노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import gc
import importlib
import os
import random
import re
import time

ANIMATE2 = {
    "animate2": "Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
    "animate2-fast": "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
}
S2V = "Wan-AI/Wan2.2-S2V-14B"
TEXT_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"  # Animate-2 저장소와 같은 UMT5 파일(sha256 같음) — 다른 Wan 플러그인과 같이 쓴다
MAX_AREA = {"720p": 1280 * 720, "480p": 832 * 480}
# Animate-2 블록은 height × width 를 넓이로만 쓰고 비율은 캐릭터 그림을 따른다(블록 설명).
ANIMATE2_SIZE = {"720p": (720, 1280), "480p": (480, 832)}
ANIMATE2_FPS = 24  # WanAnimate2Blocks fps 기본값
S2V_FPS = 16  # 공식 shared_config sample_fps

_CACHE = {"key": None, "pipe": None}


def _repo_dir(ctx, repo, probe):
    path = os.path.normpath(ctx.model_path(f"{repo}/{probe}"))
    for _ in probe.split("/"):
        path = os.path.dirname(path)
    return path


def _free():
    import torch

    _CACHE.update(key=None, pipe=None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _read_video(path):
    import av
    import numpy as np
    from PIL import Image

    frames = []
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("움직임 영상에 영상이 없습니다.")
        stream = container.streams.video[0]
        rate = float(stream.average_rate or stream.guessed_rate or 24)
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")
            rot = int(round((getattr(frame, "rotation", 0) or 0) / 90.0)) % 4
            if rot:
                arr = np.ascontiguousarray(np.rot90(arr, rot))
            frames.append(Image.fromarray(arr))
    if not frames:
        raise ValueError("움직임 영상에서 프레임을 읽지 못했습니다.")
    return frames, rate


class _SegmentProgress:
    """Animate-2 디노이즈 반복이 구간마다 여는 tqdm(total=단계 수, desc="Segment k/n") 자리 — 단계마다 취소를 보고 진행률을 알린다."""

    def __init__(self, ctx, total, desc):
        found = re.search(r"(\d+)/(\d+)", desc or "")
        self.segment, self.segments = (int(found.group(1)) - 1, int(found.group(2))) if found else (0, 1)
        self.ctx, self.total, self.done = ctx, max(1, total or 1), 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n=1):
        self.done += n
        self.ctx.check_cancel()
        part = (self.segment + min(1.0, self.done / self.total)) / max(1, self.segments)
        self.ctx.progress(0.1 + 0.85 * min(1.0, part), "generate")


def _animate2(ctx, inputs, settings, image, seed):
    import torch
    from diffusers import AutoencoderKLWan, ComponentsManager, DPMSolverMultistepScheduler, WanAnimate2Transformer3DModel
    from diffusers.modular_pipelines import WanAnimate2Blocks, WanAnimate2DistilledBlocks
    from transformers import AutoTokenizer, CLIPVisionModel, UMT5EncoderModel

    variant = settings["mode"]
    if not inputs.get("video"):
        raise ValueError("Animate-2 는 움직임 영상이 있어야 합니다. 사람이 움직이는 영상을 이어 주세요.")
    repo = ANIMATE2[variant]
    if _CACHE["key"] != variant:
        _free()
        ctx.progress(0.02, "load-model")
        root = _repo_dir(ctx, repo, "model_index.json")
        text = _repo_dir(ctx, TEXT_REPO, "text_encoder/config.json")
        blocks = WanAnimate2DistilledBlocks() if variant == "animate2-fast" else WanAnimate2Blocks()
        manager = ComponentsManager()
        pipe = blocks.init_pipeline(components_manager=manager)
        pipe.update_components(
            tokenizer=AutoTokenizer.from_pretrained(os.path.join(text, "tokenizer")),
            text_encoder=UMT5EncoderModel.from_pretrained(os.path.join(text, "text_encoder"), torch_dtype=torch.bfloat16),
            image_encoder=CLIPVisionModel.from_pretrained(os.path.join(root, "image_encoder"), torch_dtype=torch.bfloat16),
            vae=AutoencoderKLWan.from_pretrained(os.path.join(root, "vae"), torch_dtype=torch.float32),
            transformer=WanAnimate2Transformer3DModel.from_pretrained(os.path.join(root, "transformer"), torch_dtype=torch.bfloat16),
            scheduler=DPMSolverMultistepScheduler.from_pretrained(os.path.join(root, "scheduler")),
        )
        manager.enable_auto_cpu_offload(device="cuda")
        _CACHE.update(key=variant, pipe=pipe)
    pipe = _CACHE["pipe"]
    frames, rate = _read_video(inputs["video"])
    ctx.check_cancel()
    ctx.progress(0.1, "generate")
    height, width = ANIMATE2_SIZE[settings["resolution"]]
    kwargs = dict(
        image=image,
        driving_video=frames,
        driving_video_fps=rate,
        height=height,
        width=width,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output="videos",
    )
    prompt = (inputs.get("prompt") or "").strip()
    if prompt:
        kwargs["prompt"] = prompt
    # diffusers 0.42 의 Animate-2 디노이즈 반복(WanAnimate2SegmentDenoiseInner)은 블록의 progress_bar 가 아니라 제 모듈의 tqdm 을
    # 바로 부른다 — 그 이름을 이 실행 동안만 바꿔 끼운다.
    denoise = importlib.import_module("diffusers.modular_pipelines.wan_animate_2.denoise")
    original = denoise.tqdm
    denoise.tqdm = lambda iterable=None, total=None, desc="", **_: _SegmentProgress(ctx, total, desc)
    try:
        videos = pipe(**kwargs)
    finally:
        denoise.tqdm = original
    return videos[0], ANIMATE2_FPS, None


def _s2v(ctx, inputs, settings, image_path, seed):
    import torch

    from .wan.configs.wan_s2v_14B import s2v_14B
    from .wan.speech2video import WanS2V

    if not inputs.get("audio"):
        raise ValueError("S2V(말하는 영상)는 목소리 소리 파일이 있어야 합니다. 소리를 이어 주세요.")
    prompt = (inputs.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("S2V 는 프롬프트가 있어야 합니다. 인물 · 장면 · 말하는 모습을 영어로 자세히 적어 주세요.")
    if _CACHE["key"] != "s2v":
        _free()
        ctx.progress(0.02, "load-model")
        root = _repo_dir(ctx, S2V, "config.json")
        pipe = WanS2V(config=s2v_14B, checkpoint_dir=root, device_id=0, rank=0, t5_cpu=True, init_on_cpu=True, convert_model_dtype=True)
        _CACHE.update(key="s2v", pipe=pipe)
    pipe = _CACHE["pipe"]
    # 공식 generate 에는 단계 콜백이 없어 변환기(noise_model) forward 마다 취소를 보고 진행률을 알린다. 한 단계에 forward 가
    # guidance 로 두 번(조건 · 무조건)이고, 클립 수는 소리 길이로 generate 안에서 정해져(num_repeat=None) encode_audio 가 돌려준 수를 받아 둔다.
    calls_per_clip = s2v_14B.sample_steps * (2 if s2v_14B.sample_guide_scale > 1 else 1)
    state = {"clips": 1, "calls": 0}
    encode_audio = pipe.encode_audio

    def count_clips(*args, **kwargs):
        emb, clips = encode_audio(*args, **kwargs)
        state["clips"] = max(1, int(clips))
        return emb, clips

    def on_forward(_module, _args):
        ctx.check_cancel()
        state["calls"] += 1
        ctx.progress(0.1 + 0.85 * min(1.0, state["calls"] / (calls_per_clip * state["clips"])), "generate")

    pipe.encode_audio = count_clips
    handle = pipe.noise_model.register_forward_pre_hook(on_forward)
    ctx.progress(0.1, "generate")
    try:
        video = pipe.generate(
            input_prompt=prompt,
            ref_image_path=image_path,
            audio_path=inputs["audio"],
            enable_tts=False,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            num_repeat=None,
            pose_video=None,
            max_area=MAX_AREA[settings["resolution"]],
            infer_frames=80,
            shift=s2v_14B.sample_shift,
            sample_solver="unipc",
            sampling_steps=s2v_14B.sample_steps,
            guide_scale=s2v_14B.sample_guide_scale,
            n_prompt="",
            seed=seed,
            offload_model=True,
            init_first_frame=False,
        )
    finally:
        handle.remove()
        del pipe.encode_audio  # 덮어쓴 것을 지워 클래스 메서드로 되돌린다
        # 취소 · 오류로 단계 중에 멈추면 변환기가 그래픽카드에 남는다(끝까지 돌면 generate 가 이미 내렸다).
        pipe.noise_model.cpu()
        torch.cuda.empty_cache()
    # (C, T, H, W) -1~1 → (T, H, W, C) 0~1
    frames = ((video.float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).cpu().numpy()
    return frames, S2V_FPS, inputs["audio"]


def wan_a2s(ctx, inputs, settings):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    t0 = time.monotonic()
    if ctx.device != "cuda":
        raise ValueError("Wan Animate-2 · S2V 는 NVIDIA 그래픽카드(CUDA)에서만 돕니다.")
    with Image.open(inputs["image"]) as im:
        im.load()
        image = ImageOps.exif_transpose(im).convert("RGB")
    seed = int(settings["seed"])
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    try:
        if settings["mode"] == "s2v":
            frames, fps, audio_path = _s2v(ctx, inputs, settings, inputs["image"], seed)
        else:
            frames, fps, audio_path = _animate2(ctx, inputs, settings, image, seed)
        ctx.check_cancel()
        ctx.progress(0.95, "encode")
        frames = np.asarray(frames, dtype=np.float32)
        out = ctx.out_path("wan-a2s.mp4")
        _write_mp4(out, frames, fps, audio_path)
        ctx.log(f"done in {time.monotonic() - t0:.1f}s")
    except torch.OutOfMemoryError:
        oom = True
    else:
        oom = False
    if oom:
        # except 블록 밖에서 정리하고 새 오류를 낸다 — 블록 안에서 내면 원래 오류(와 그 프레임이 쥔 그래픽카드 텐서)가
        # 새 오류의 __context__ 로 남아 사이드카가 다음 실행까지 메모리를 놓지 못한다.
        _free()
        raise RuntimeError("그래픽카드 메모리가 모자라 멈췄습니다. 이 모델은 서버급 그래픽카드(32GB 초과)가 필요합니다.")
    ctx.progress(1.0, "done")
    return {"video": out}


def _write_mp4(path, frames, fps, audio_path):
    """frames (n, h, w, 3) 0~1 → H.264 yuv420p mp4. audio_path 가 있으면 그 소리를 AAC 로 넣고 영상 길이에 맞춰 자른다."""
    import av
    import numpy as np

    n, h, w, _ = frames.shape
    h2, w2 = h - h % 2, w - w % 2
    container = av.open(path, "w", options={"movflags": "+faststart"})
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = w2, h2
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16", "preset": "medium"}
        astream = container.add_stream("aac", rate=44100) if audio_path else None  # 머리 정보를 쓰기 전에 두 흐름을 다 만든다
        for i in range(n):
            arr = (np.clip(frames[i, :h2, :w2], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        if astream is not None:
            duration = n / fps
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=44100)
            with av.open(audio_path) as src:
                for frame in src.decode(audio=0):
                    if frame.time is not None and frame.time > duration:
                        break
                    for rf in resampler.resample(frame):
                        for packet in astream.encode(rf):
                            container.mux(packet)
            for packet in astream.encode(None):
                container.mux(packet)
    finally:
        container.close()


NODES = {"wan-a2s": wan_a2s}
