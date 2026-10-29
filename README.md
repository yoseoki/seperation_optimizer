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

모델은 CPU 또는 CUDA device에서 학습할 수 있습니다. 다만 subspace snapshot은 대표 weight를 CPU `float64` Tensor로 복사한 뒤 SVD로 계산하므로, 큰 모델에서는 GPU에서 CPU로의 전송과 CPU SVD가 병목이 될 수 있습니다.

현재 구현은 CuPy를 사용하거나 요구하지 않습니다.

## 테스트

```bash
python -m pytest
```
