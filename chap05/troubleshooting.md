# 🎙️ 오디오 텍스트 변환 프로젝트 트러블슈팅 가이드

## 📋 프로젝트 개요
Whisper 모델을 활용하여 로컬 환경에서 오디오 파일을 텍스트로 변환하는 프로젝트 진행 중 발생한 주요 이슈 및 해결 방법 정리.

---

## 🔴 발생한 주요 문제들

### 1️⃣ **FFmpeg 경로 설정 미반영**

#### ❌ 문제 상황
```
ValueError: ffmpeg was not found but is required to load audio files from filename
```

#### 🔍 원인 분석
- `os.environ["PATH"]`로 FFmpeg 경로를 설정했으나 Python 프로세스가 인식하지 못함
- 이미 실행 중인 프로세스는 환경변수를 다시 읽지 않음
- FFmpeg 설정이 이루어지기 **전에** 모델 로드 코드가 실행됨

#### ✅ 해결 방법

**Step 1: 별도의 셀에서 먼저 실행**
```python
import os

# FFmpeg 경로를 먼저 설정 (대문자 PATH 주의!)
os.environ["PATH"] += os.pathsep + r"C:\Users\hoyon\OneDrive\바탕 화면\개발공부\LLM을활용한AI에이전트\chap05\ffmpeg-2025-12-07-git-c4d22f2d2c-full_build\bin"

print("✓ FFmpeg 경로 설정 완료")
```

**Step 2: 그 다음 셀에서 모델 로드**
```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cpu"  # 또는 "cuda:0" if torch.cuda.is_available()
torch_dtype = torch.float32  # 또는 torch.float16 (GPU 사용 시)

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,
    chunk_length_s=10,
    stride_length_s=2,
)

print("✓ 모델 및 파이프라인 로드 완료")
```

#### 💡 핵심 포인트
- **`PATH`는 대문자** (소문자 "path" 사용 시 인식 안 됨)
- **반드시 모델 로드 전에 설정**
- **셀 순서가 중요함** (FFmpeg 설정 셀 → 모델 로드 셀)

---

### 2️⃣ **CUDA 버전 패키지 설치 실패**

#### ❌ 문제 상황
```
✓ torch 버전: 2.9.1+cpu  # CPU 버전으로 설치됨
✓ CUDA 사용 가능: False
```

#### 🔍 원인 분석
- PyTorch를 기본 설치 명령어로 설치하면 CPU 버전만 설치됨
- GPU를 사용하려면 CUDA 지원 버전을 명시적으로 지정해야 함
- 잘못된 pip 명령어로 설치하면 호환성 문제 발생

#### ✅ 해결 방법

**GPU 있는 경우:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**GPU 없는 경우:**
```bash
pip install torch torchvision torchaudio
```

#### 💡 핵심 포인트
- GPU 없으면 무리해서 CUDA 설치할 필요 없음 (CPU로도 충분)
- Jupyter 노트북 환경은 주로 CPU 사용
- 설치 후 **반드시 커널 재시작** 필요

---

### 3️⃣ **Torch와 Torchvision 버전 불일치**

#### ❌ 문제 상황
```
RuntimeError: operator torchvision::nms does not exist
```

#### 🔍 원인 분석
- torch와 torchvision의 빌드 버전이 호환되지 않음
- torch는 CPU 버전, torchvision은 CUDA 버전 등 섞여 설치됨
- 기존 캐시 파일이 남아있어 버전 충돌 발생

#### ✅ 해결 방법

**Step 1: 모든 torch 관련 패키지 제거**
```bash
pip uninstall -y torch torchvision torchaudio
pip cache purge
```

**Step 2: 올바른 버전으로 재설치**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Step 3: Jupyter 커널 재시작**
```
Kernel → Restart Kernel and Clear All Outputs
```

#### 💡 핵심 포인트
- 완전 제거 후 재설치가 필수
- 캐시 제거 중요
- 버전 호환성은 항상 확인 필요

---

### 4️⃣ **transformers 모듈 import 오류**

#### ❌ 문제 상황
```
ModuleNotFoundError: Could not import module 'AutoProcessor'
ImportError: cannot import name 'add_model_info_to_auto_map'
```

#### 🔍 원인 분석
- transformers 라이브러리 내부 모듈들의 버전 불일치
- 이전 버전의 캐시 파일이 남아있음
- 패키지 버전이 명시되지 않아 호환되지 않는 버전이 설치됨

#### ✅ 해결 방법

**Step 1: transformers 캐시 제거 및 업그레이드**
```bash
pip uninstall -y transformers
pip cache purge
pip install --upgrade transformers
```

**Step 2: Jupyter 커널 재시작**
```
Kernel → Restart Kernel and Clear All Outputs
```

#### 💡 핵심 포인트
- 버전 명시 없으면 최신 안정 버전 설치 권장
- 라이브러리 내부 불일치는 완전 제거 후 재설치로 해결
- 주기적인 커널 재시작 필수

---

## 🎓 CUDA 개념 정리

### CUDA란?
- **CUDA** = Compute Unified Device Architecture
- NVIDIA GPU를 활용하여 병렬 연산을 수행하기 위한 기술
- GPU를 사용하면 CPU보다 훨씬 빠른 연산 가능

### 당신의 상황
```
Jupyter 노트북 환경 → GPU 없음 → CUDA 불필요
↓
CPU로 충분히 작동 가능
```

### 확인 방법
```python
import torch
print(torch.cuda.is_available())  # False = GPU 없음, True = GPU 있음
```

### CUDA 관련 설정
```python
# GPU 있으면 사용, 없으면 CPU 사용
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# float16은 GPU 연산 최적화, float32는 CPU 표준
```

---

## 📦 최종 설정 가이드

### 권장 설치 순서

#### 1단계: 기본 패키지
```bash
pip install --upgrade pip
pip install transformers datasets[audio] accelerate
```

#### 2단계: PyTorch (CPU 버전 권장)
```bash
pip install torch torchvision torchaudio
```

#### 3단계: 오디오 처리 라이브러리
```bash
pip install librosa soundfile
```

#### 4단계: Jupyter 도구
```bash
pip install jupyter ipywidgets
```

### Jupyter 노트북 최적 구조

```python
# [셀 1] 패키지 임포트
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# [셀 2] FFmpeg 경로 설정 (반드시 먼저!)
os.environ["PATH"] += os.pathsep + r"C:\Users\...\ffmpeg\bin"

# [셀 3] 모델 로드
device = "cpu"
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(...)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(...)

# [셀 4] 오디오 처리
result = pipe("./audio/file.mp3")
print(result["text"])
```

---

## ✅ 체크리스트

프로젝트 시작 전 다음을 확인하세요:

- [ ] FFmpeg 설치 및 경로 확인
- [ ] 패키지 설치 순서 준수 (transformers → torch → librosa)
- [ ] Jupyter 커널 재시작 완료
- [ ] `torch.cuda.is_available()` 확인 (False여도 정상)
- [ ] FFmpeg 경로 설정 코드가 모델 로드 전에 실행됨
- [ ] 오디오 파일 경로 정확성 확인

---

## 🚀 빠른 해결 플로우

### 문제: "ffmpeg was not found"
```
→ Step 1: os.environ["PATH"] 코드 실행
→ Step 2: 모델 로드 코드 실행
```

### 문제: "CUDA 관련 오류"
```
→ Step 1: CUDA 설치 불필요 (GPU 없으면)
→ Step 2: CPU 버전으로 설치 후 진행
```

### 문제: "import 오류"
```
→ Step 1: pip uninstall -y [패키지명]
→ Step 2: pip cache purge
→ Step 3: pip install [패키지명]
→ Step 4: Kernel → Restart
```

---

## 📚 참고 자료

- [PyTorch 공식 문서](https://pytorch.org/)
- [Transformers 설치 가이드](https://huggingface.co/docs/transformers/installation)
- [Jupyter Notebook 커널 문제 해결](https://jupyter.readthedocs.io/en/latest/)

---

