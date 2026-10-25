# Wan Animate-2 · S2V — 동작 옮기기 · 말하는 영상 (서버급)

PromBoard 커뮤니티 플러그인입니다. 캔버스에 **Wan Animate-2 · S2V** 노드 하나를 더하고, 알리바바 Wan 팀의 공개 모델 둘을 모드로 고릅니다.

- **Animate-2**(Wan2.2-Animate-2-14B, 2026-07): 캐릭터 그림 속 인물이 움직임 영상 속 사람처럼 움직이는 영상. 관절을 따로 뽑지 않고 움직임 영상을 그대로 받습니다. 빠른 판(증류, 10단계)과 일반 판(40단계).
- **S2V 말하는 영상**(Wan2.2-S2V-14B): 캐릭터 그림 + 목소리 파일 + 프롬프트로 그 소리에 맞춰 말하는 영상. 결과에 입력 소리가 들어가고, 길이는 소리를 따릅니다.

> 🔴 **미검증 · 서버급**: 실제로 돌려 보지 않았습니다. S2V 공식 안내가 그래픽카드 80GB 이상이고 Animate-2 공식 시험 환경도 A800 여러 장이라, 소비자 그래픽카드(최대 32GB)로는 돌지 않는다고 보고 PromBoard 에 '서버급'으로 보이게 했습니다. 문제가 있으면 [이슈](https://github.com/promboard-app/wan-animate2-s2v/issues)로 알려 주세요.

## 입력과 결과

| 포트 | 종류 | 쓰는 모드 |
|---|---|---|
| 캐릭터 그림 | 그림(필수) | 둘 다 |
| 움직임 영상 | 영상 | Animate-2(필수) |
| 목소리 | 소리 | S2V(필수) |
| 프롬프트 | 글 | S2V(필수), Animate-2(선택) |
| 영상 | 영상 | Animate-2: 24fps · 소리 없음 / S2V: 16fps · 입력 소리 포함 |

- 설정: 모드(Animate-2 빠른 판 · 일반 판 · S2V), 해상도(720P · 480P, 넓이 기준 — 화면 비는 캐릭터 그림을 따른다), 시드.
- S2V 생성 값은 공식 `generate.py` 기본값입니다: 클립당 80프레임, 40단계, guidance 4.5, shift 3, UniPC.

## 받는 것과 용량 (고른 모드의 파일만, 처음 실행할 때 한 번)

| 모드 | 파일 | 크기 |
|---|---|---|
| Animate-2 빠른 판 | [Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers) 변환기 · CLIP 그림 인코더 · VAE + 글 인코더 UMT5-XXL([Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers), 같은 파일 — 다른 Wan 플러그인과 같이 씀) | 약 46GB |
| Animate-2 일반 판 | [Wan-AI/Wan2.2-Animate-2-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Diffusers) + 같은 글 인코더 | 약 46GB |
| S2V | [Wan-AI/Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) 변환기 · T5 · VAE · wav2vec2 | 약 46GB |

## 라이선스와 조건

- 플러그인 코드: Apache-2.0([LICENSE](LICENSE)).
- `nodes/wan/` 는 Wan2.2 공식 코드(Apache-2.0, [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) `wan/speech2video.py` 와 그 모듈)를 옮긴 것입니다. diffusers 에 S2V 파이프라인이 없어서입니다. 바꾼 곳은 `PromBoard:` 주석으로 표시했습니다 — flash-attn 이 없을 때 같은 계산을 PyTorch SDPA 로 하는 것(PromBoard 는 바이너리 휠만 깔아 flash-attn 을 못 깝니다), decord 를 쓰는 곳(이어 만들기 · 포즈 영상, 이 노드는 쓰지 않음)만 그때 불러오는 것, 쓰지 않는 모듈까지 불러오는 패키지 `__init__` 를 비운 것.
- Animate-2 는 diffusers 0.42 Modular Diffusers 블록(`WanAnimate2Blocks` · `WanAnimate2DistilledBlocks`)을 씁니다. 저장소의 `modular_model_index.json` 은 허깅페이스 주소를 가리켜 오프라인에서 읽을 수 없어, 부품을 받은 파일에서 직접 읽어 채웁니다.
- 모델 가중치: Apache-2.0. 매출 상한 · 지역 제한 · 사용 승인 절차가 없습니다.

## 설치

1. PromBoard 앱 → 플러그인 매니저 → **커뮤니티 플러그인** → 둘러보기에서 설치합니다.
2. 코드가 든 플러그인이라 **제한 모드**를 끄고 플러그인을 켭니다.
3. 처음 실행할 때 고른 모드의 모델(약 46GB)과 파이썬 패키지를 받습니다.

---

## English

A PromBoard community plugin. It adds one **Wan Animate-2 · S2V** node with two of Alibaba's open Wan models as modes:

- **Animate-2** (Wan2.2-Animate-2-14B, July 2026): the person in the character image moves like the person in the driving video. It takes the driving video directly, no pose extraction. Fast (distilled, 10 steps) and normal (40 steps) variants.
- **S2V talking video** (Wan2.2-S2V-14B): character image + voice audio + prompt → a video that talks along with the audio. The input audio is muxed into the result and the length follows the audio.

> 🔴 **Unverified · server-class**: not run yet. The official S2V guidance asks for 80 GB+ of VRAM and Animate-2 was tested on multiple A800s, so it is marked server-class in PromBoard (more than the largest consumer card, 32 GB). Please report problems in [issues](https://github.com/promboard-app/wan-animate2-s2v/issues).

- Inputs: character image (required), driving video (Animate-2), voice (S2V), prompt (required for S2V). Output: one MP4 (Animate-2 24 fps without audio, S2V 16 fps with the input audio). Settings: mode, resolution by area (720P · 480P, aspect follows the character image), seed. S2V uses the official `generate.py` defaults (80 frames per clip, 40 steps, guidance 4.5, shift 3, UniPC).
- Downloads (only the chosen mode, once): about 46 GB each — Animate-2 (Distilled-)Diffusers transformer · CLIP image encoder · VAE plus the UMT5-XXL text encoder from Wan-AI/Wan2.2-TI2V-5B-Diffusers (same file, shared with the other Wan plugins), or Wan-AI/Wan2.2-S2V-14B.
- `nodes/wan/` vendors the official Wan2.2 S2V code (Apache-2.0) because diffusers has no S2V pipeline. Changes are marked `PromBoard:` — an SDPA fallback with the same varlen semantics when flash-attn is missing (PromBoard installs binary wheels only), lazy decord imports, and emptied package `__init__` files. Animate-2 fills the diffusers 0.42 Modular Diffusers blocks from the downloaded files, because the repo's `modular_model_index.json` points at Hub revisions that cannot be read offline.
- **Install**: PromBoard → Plugin Manager → **Community plugins** → Browse → install. Turn off **Restricted mode** and enable the plugin.

The plugin texts in the app are Korean only for now — the plugin manifest takes a single language.

## License

Plugin code: Apache-2.0 — see [LICENSE](LICENSE). `nodes/wan/` is official Wan2.2 code (Apache-2.0, [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)). The model weights are Apache-2.0 with no revenue cap, regional restriction or access gate, and are downloaded from Hugging Face, not bundled.
