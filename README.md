# Seperation Optimizer

`seperation-optimizer`는 모델 block별 weight subspace의 변화를 이용해 내부 SGD의 learning rate를 조정하는 PyTorch optimizer입니다. 공개 클래스 이름 `SeperationOptimizer`는 기존 API에서 요구한 철자를 그대로 사용합니다.

## 주요 기능

- 모델 구성과 block metadata를 사용자 구현 `ModelBlockInfo`로 주입합니다.
- block별 고정 배수를 사용자 구현 `AlphaProvider`로 주입합니다.
- 각 대표 레이어의 최근 subspace snapshot 세 개만 유지합니다.
- 기준 acceleration magnitude와 현재 magnitude의 비율을 block별 learning rate에 반영합니다.
- 내부 `torch.optim.SGD`에 momentum과 weight decay를 전달합니다.
- 모델 이름이나 모델별 registry에 의존하지 않습니다.

## 설치

저장소에서 개발 모드로 설치하려면 다음 명령을 사용합니다.

```bash
python -m pip install -e ".[test]"
```

일반 설치는 다음과 같습니다.

```bash
python -m pip install .
```

## 기본 사용법

먼저 모델의 생성 방식과 ordered block metadata를 제공하는 `ModelBlockInfo`를 구현합니다. `get_block_info()`가 반환하는 대표 이름과 파라미터 그룹은 반드시 같은 길이와 순서를 가져야 합니다.

```python
import torch
import torch.nn as nn

from seperation_optimizer import AlphaProvider
from seperation_optimizer import ModelBlockInfo
from seperation_optimizer import SeperationOptimizer


class MyModelBlockInfo(ModelBlockInfo):

    def build_model(self, num_classes):
        """작은 분류 모델을 생성한다. 입력은 int이고 출력은 Module이다. num_classes는 출력 클래스 수를 의미한다."""
        return nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def get_non_save_layers(self):
        """대표 weight에서 제외할 이름 목록을 반환한다. 입력은 없고 출력은 str list이다. bias는 대표 파라미터 선택에서 제외된다."""
        return ["bias"]

    def get_block_info(self, model):
        """대표 이름과 파라미터 그룹을 순서대로 반환한다. 입력은 Module이고 출력은 두 list의 tuple이다. 같은 위치의 이름과 그룹은 동일한 block을 의미한다."""
        names = ["0.weight", "2.weight"]
        groups = [
            list(model[0].parameters()),
            list(model[2].parameters()),
        ]
        return names, groups


class MyAlphaProvider(AlphaProvider):

    def get_alphas(self):
        """block별 고정 alpha를 반환한다. 입력은 없고 출력은 float list이다. 값의 순서는 block 정보의 순서와 일치한다."""
        return [1.0, 0.5]
```

구현체로 모델과 optimizer를 생성합니다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

block_info = MyModelBlockInfo()
alpha_provider = MyAlphaProvider()
model = block_info.build_model(num_classes=10).to(device)

optimizer = SeperationOptimizer(
    model,
    block_info,
    alpha_provider,
    base_lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    epoch_offset=3,
    epoch_interval=5,
    reshape_mode="in_dim",
)

criterion = nn.CrossEntropyLoss()
```

`step_lr(epoch)`은 각 epoch의 batch loop 직전에 정확히 한 번 호출합니다.

```python
for epoch in range(100):
    optimizer.step_lr(epoch)

    model.train()
    for inputs, targets in trainloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
```

## Epoch 정책

`step_lr(epoch)`은 다음 순서로 동작합니다.

1. 호출할 때마다 현재 대표 weight의 subspace snapshot을 추가합니다.
2. `epoch == epoch_offset - 1`이면 최근 snapshot 세 개로 기준 magnitude를 설정합니다.
3. `epoch >= epoch_offset`이고 interval 조건을 만족하면 magnitude와 alpha를 결합해 block별 learning rate를 갱신합니다.

기준점 설정에는 snapshot 세 개가 필요하므로 일반적으로 `epoch_offset >= 3`으로 설정해야 합니다. snapshot이 부족하면 `RuntimeError`가 발생합니다.

## Block metadata 계약

다음 순서는 모두 일치해야 합니다.

1. 모델의 대표 파라미터 순회 순서
2. `get_block_info()`의 대표 이름 순서
3. `get_block_info()`의 파라미터 그룹 순서
4. `get_alphas()`의 alpha 순서

대표 이름은 모델의 `named_parameters()`에 있는 정확한 이름이거나 하나의 파라미터만 선택하는 prefix여야 합니다. 대표 파라미터는 같은 위치의 파라미터 그룹에 포함되어야 하며, 내부 SGD는 그룹에 포함된 파라미터만 갱신합니다.

## Reshape mode

지원하는 `reshape_mode`는 다음과 같습니다.

- `in_dim`: convolution weight의 input-channel 관점 subspace를 계산합니다.
- `out_dim`: convolution weight의 output-channel 관점 subspace를 계산합니다.
- `ker_dim`: convolution kernel 차원 관점 subspace를 계산합니다.
- `flatten`: 대표 weight 전체를 정규화된 하나의 vector로 처리합니다.

2차원 linear weight는 행과 열 크기를 기준으로 작은 차원을 표본 축으로 사용합니다.

## Weight decay와 scheduler

`weight_decay`는 내부 `torch.optim.SGD`의 표준 coupled weight decay로 정상 적용됩니다.

```python
optimizer = SeperationOptimizer(
    model,
    block_info,
    alpha_provider,
    base_lr=0.01,
    weight_decay=1e-4,
)
```

현재 `SeperationOptimizer`는 `torch.optim.Optimizer`를 상속하지 않으므로 PyTorch의 표준 learning-rate scheduler에 직접 전달할 수 없습니다.

```python
# 현재 지원하지 않음
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
```

내부 `_optimizer`에 scheduler를 연결하는 것도 권장하지 않습니다. subspace 갱신 epoch에 learning rate가 다시 `base_lr * magnitude * alpha`로 설정되어 scheduler 결과와 충돌할 수 있습니다.

## CPU와 CUDA

CUDA 모델의 subspace snapshot, SVD, 투영과 magnitude 계산은 CuPy로 GPU에서 수행합니다. PyTorch와 CuPy는 DLPack으로 GPU 메모리를 공유하며, snapshot은 별도로 복사해 이후 weight 갱신의 영향을 받지 않습니다. 행렬 전체를 CPU로 전송하지 않지만 rank 판정과 학습률 계산에 필요한 scalar는 CPU로 가져옵니다. snapshot은 모델과 같은 GPU의 float64 Tensor로 보관됩니다.

CPU 모델은 기존 PyTorch CPU 경로를 사용하며 CuPy가 필요하지 않습니다. CUDA 모델에 CuPy가 없으면 설치 안내와 함께 오류가 발생합니다. 모델을 GPU로 옮기는 것은 사용자가 `model.to("cuda")`로 수행해야 하며, snapshot 수집 후에는 모델 device를 변경하지 마세요.

### 실행 시 CuPy 자동 선택

CUDA용 PyTorch와 CuPy를 설치해도 모델과 입력이 자동으로 GPU로 이동하지는 않습니다. 사용자가 모델을 CUDA device로 옮긴 뒤 optimizer를 생성하고, 학습에 사용하는 입력과 target도 같은 device로 옮겨야 합니다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = block_info.build_model(num_classes=10).to(device)
optimizer = SeperationOptimizer(
    model, block_info, alpha_provider, base_lr=0.01
)

for epoch in range(100):
    optimizer.step_lr(epoch)
    model.train()
    for inputs, targets in trainloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
```

모듈은 부분공간을 계산할 대표 파라미터의 device를 확인합니다. 파라미터가 CUDA에 있으면 별도 옵션 없이 CuPy를 사용하고, CPU에 있으면 PyTorch CPU 연산을 사용합니다. 입력 데이터의 device는 이 선택의 기준이 아닙니다. 모델의 forward/backward와 SGD 갱신은 PyTorch가 수행하고, CuPy는 부분공간 행렬 연산을 수행합니다.

### GPU 환경 설치

**`pip install .` 또는 `pip install -e .`는 `scripts/install.py`를 자동 실행하지 않습니다.** GPU 감지와 그에 따른 의존성 설치를 사용하려면 저장소에서 `python scripts/install.py`를 직접 실행하세요. 이 스크립트는 PyTorch와 필요한 CuPy를 설치한 다음 현재 패키지를 개발 모드로 설치합니다.

`pyproject.toml` 의존성은 하드웨어를 감지하거나 PyTorch wheel 저장소를 지정할 수 없습니다. 다음 스크립트는 `nvidia-smi`로 GPU 접근 가능 여부를 확인해 GPU용 PyTorch와 CuPy 또는 CPU용 PyTorch를 설치합니다. **현재 Python 환경의 PyTorch를 재설치하므로 전용 가상환경에서 실행하세요.**

```bash
# 먼저 실제 실행할 명령 확인
python scripts/install.py --dry-run
# GPU 감지 시 CUDA 12.6 PyTorch와 cupy-cuda12x 설치
python scripts/install.py
# 드라이버와 GPU에 맞는 버전을 명시
python scripts/install.py --device cuda --cuda 12.8
# CPU 환경 강제 선택
python scripts/install.py --device cpu
```

지원 선택지는 CUDA 12.6, 12.8, 13.0입니다. 자동 감지는 드라이버 호환 버전이나 GPU 아키텍처까지 판정하지 않으므로 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에서 환경에 맞는 버전을 확인해 `--cuda`로 지정하세요. 드라이버 접근이 불가능하면 auto 모드는 CPU를 선택합니다.

수동 설치 예시(CUDA 12.6):

```bash
python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[cuda12]"
```

CUDA 13 계열에는 `[cuda13]` extra를 사용합니다. 서로 다른 CuPy 배포판을 함께 설치하지 마세요. 자세한 CUDA 요구사항은 [CuPy 설치 안내](https://docs.cupy.dev/en/stable/install.html)를 참고하세요.

## 테스트

```bash
python -m pytest
```
