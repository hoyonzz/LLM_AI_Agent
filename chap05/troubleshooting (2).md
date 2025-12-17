# 🔥 트러블슈팅: Windows에서 Pyannote.audio 설치 지옥 탈출기 (2025년 기준)

## 1. 배경 (Background)
Windows 환경에서 화자 분리(Speaker Diarization)를 위해 `pyannote.audio 3.1` 모델을 사용하려 했으나, 패키지 간 버전 충돌(Dependency Hell)과 PyTorch 보안 정책 변경으로 인해 실행 불가 현상 발생.

*   **OS:** Windows 10/11
*   **Python:** 3.12
*   **Goal:** `pyannote/speaker-diarization-3.1` 모델 구동

## 2. 발생했던 주요 오류 (Symptoms)

### 🛑 1) 무한 로딩 (Deadlock)
*   **현상:** 에러 메시지 없이 코드가 멈춤 (Pending).
*   **원인:** Windows 환경에서 Numpy와 PyTorch가 OpenMP를 중복 호출하며 충돌.

### 🛑 2) Torchvision & Torchaudio 버전 불일치
*   **현상:** `AttributeError: partially initialized module 'torchvision' ...`
*   **원인:** `pip install` 시 서로 호환되지 않는 버전이 뒤섞임.

### 🛑 3) PyTorch 2.6.0 보안 이슈
*   **현상:** `UnpicklingError` 또는 `WeightsUnpickler error`.
*   **원인:** PyTorch 2.6부터 `weights_only=True`가 기본값이 되면서, 기존 Pyannote 모델 로딩을 차단함.

### 🛑 4) Huggingface_hub 파라미터 오류
*   **현상:** `TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'`
*   **원인:** `huggingface_hub` 최신 버전(0.27+)에서 `use_auth_token` 파라미터 삭제됨.

---

## 3. 해결 방법 (Solution)

결론적으로 **가장 안정적인 "황금 버전 조합(Golden Combination)"**으로 버전을 고정(Pinning)하여 해결함.

### ✅ Step 1: 기존 패키지 완전 삭제
터미널에서 가상환경(venv) 활성화 후 실행:
```bash
pip uninstall torch torchvision torchaudio pyannote.audio lightning numpy huggingface_hub -y
```

### ✅ Step 2: 호환성 검증된 버전 설치 (핵심 ⭐)
*   **Torch:** 2.5.1 (보안 이슈 없는 마지막 안정 버전)
*   **Pyannote:** 3.3.1 (Torch 2.x 지원)
*   **Numpy:** 2.0 미만 (1.x 버전 유지)
*   **Huggingface Hub:** 0.27 미만

```bash
pip install "torch==2.5.1" "torchaudio==2.5.1" "torchvision==0.20.1" "pyannote.audio==3.3.1" "numpy<2.0" "huggingface_hub<0.27"
```

### ✅ Step 3: 실행 코드 작성
Windows 환경 변수 설정(`KMP_DUPLICATE_LIB_OK`)이 필수적임.

```python
import os
import torch
from pyannote.audio import Pipeline

# [필수] Windows OpenMP 중복 에러 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 설치 버전 확인
print(f"Torch Version: {torch.__version__}")  # 2.5.1+cpu 예상

# 파이프라인 로드
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HUGGINGFACE_TOKEN_HERE"
)

# 화자 분리 실행
# AUDIO_FILE = "path/to/your/audio.mp3"
# diarization = pipeline(AUDIO_FILE)
# ...
```

## 4. 결론 (Conclusion)
*   최신 버전(Torch 2.6, Numpy 2.0)이 항상 정답은 아님.
*   라이브러리 간의 의존성 충돌 시, **안정적인 구버전(Stable Version)**으로 롤백(Downgrade)하는 것이 정신 건강에 이로움.
*   Windows에서는 `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` 설정이 거의 필수.