from collections import deque

import pytest
import torch
import torch.nn as nn

from seperation_optimizer import seperation_optimizer as buffer_module


class TinyBlockModel(nn.Module):
    """두 블록으로 구성된 계약 검증용 모델이다."""

    def __init__(self):
        """두 개의 선형 블록을 초기화한다. 입력은 없고 출력은 초기화된 TinyBlockModel 인스턴스이다. 두 블록은 파라미터 그룹의 순서와 갱신을 검증하는 데 사용된다."""
        super().__init__()
        self.block_a = nn.Linear(2, 2, bias=False)
        self.block_b = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.block_a.weight.fill_(1.0)
            self.block_b.weight.fill_(1.0)

    def forward(self, inputs):
        """입력을 두 선형 블록에 차례로 전달한다. 입력은 torch.Tensor이고 출력은 torch.Tensor이다. 출력은 두 블록이 모두 학습 경로에 참여한 예측값을 의미한다."""
        return self.block_b(self.block_a(inputs))


class FakeModelBlockInfo(buffer_module.ModelBlockInfo):
    """작은 모델의 block 정보를 제공하는 사용자 구현체이다."""

    def build_model(self, num_classes):
        """계약 테스트에 사용할 작은 모델을 생성한다. 입력은 int이고 출력은 TinyBlockModel이다. num_classes는 실제 사용자 구현의 모델 생성 인자를 나타낸다."""
        return TinyBlockModel()

    def get_non_save_layers(self):
        """저장과 분석에서 제외할 레이어 이름을 반환한다. 입력은 없고 출력은 str 원소를 갖는 list이다. 반환값은 대표 레이어 선택에서 제외할 이름 목록을 의미한다."""
        return ["bias"]

    def get_block_info(self, model):
        """모델의 대표 이름과 파라미터 그룹을 순서대로 반환한다. 입력은 nn.Module이고 출력은 두 list를 담은 tuple이다. 두 list의 같은 위치는 동일한 모델 block을 의미한다."""
        names = ["block_a.weight", "block_b.weight"]
        groups = [list(model.block_a.parameters()), list(model.block_b.parameters())]
        return names, groups


class FakeAlphaProvider(buffer_module.AlphaProvider):
    """두 block의 고정 alpha를 제공하는 사용자 구현체이다."""

    def get_alphas(self):
        """block 순서에 대응하는 alpha 목록을 반환한다. 입력은 없고 출력은 float 원소를 갖는 list이다. 각 값은 해당 block의 magnitude에 곱할 고정 배수를 의미한다."""
        return [2.0, 3.0]


class MismatchedModelBlockInfo(FakeModelBlockInfo):
    """이름과 그룹 개수가 다른 잘못된 사용자 구현체이다."""

    def get_block_info(self, model):
        """개수가 일치하지 않는 block 정보를 반환한다. 입력은 nn.Module이고 출력은 두 list를 담은 tuple이다. 반환값은 생성자 불변식 위반을 검증하기 위한 잘못된 정보이다."""
        names = ["block_a.weight"]
        groups = [list(model.block_a.parameters()), list(model.block_b.parameters())]
        return names, groups


class MismatchedAlphaProvider(buffer_module.AlphaProvider):
    """block 수와 다른 개수의 alpha를 제공하는 잘못된 구현체이다."""

    def get_alphas(self):
        """block 수보다 적은 alpha 목록을 반환한다. 입력은 없고 출력은 float 원소를 갖는 list이다. 반환값은 alpha 개수 불변식 위반을 검증하기 위한 값이다."""
        return [2.0]


class ReorderedModelBlockInfo(FakeModelBlockInfo):
    """모델 순회와 반대 순서의 block 정보를 제공하는 잘못된 구현체이다."""

    def get_block_info(self, model):
        """모델 파라미터 순회와 반대인 block 정보를 반환한다. 입력은 nn.Module이고 출력은 두 list를 담은 tuple이다. 반환값은 대표 이름과 그룹의 순서 불변식 위반을 검증하기 위한 정보이다."""
        names = ["block_b.weight", "block_a.weight"]
        groups = [list(model.block_b.parameters()), list(model.block_a.parameters())]
        return names, groups


@pytest.fixture
def model():
    """각 테스트에 독립적인 작은 모델을 제공한다. 입력은 없고 출력은 TinyBlockModel이다. 반환 모델은 block 계약과 SGD 갱신 검증에 사용된다."""
    return TinyBlockModel()


@pytest.fixture
def block_info():
    """정상적인 모델 block 정보 구현체를 제공한다. 입력은 없고 출력은 FakeModelBlockInfo이다. 반환 객체는 사용자 주입 계약을 검증하는 데 사용된다."""
    return FakeModelBlockInfo()


@pytest.fixture
def alpha_provider():
    """정상적인 alpha 제공자 구현체를 제공한다. 입력은 없고 출력은 FakeAlphaProvider이다. 반환 객체는 block별 학습률 배수 계약을 검증하는 데 사용된다."""
    return FakeAlphaProvider()


def build_optimizer(model, block_info, alpha_provider, **overrides):
    """공통 설정으로 SeperationOptimizer를 생성한다. 입력은 모델과 두 구현체 및 설정 dict이고 출력은 SeperationOptimizer이다. 반환 optimizer는 각 계약 테스트의 검증 대상이다."""
    settings = {
        "base_lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "epoch_offset": 3,
        "epoch_interval": 2,
    }
    settings.update(overrides)
    return buffer_module.SeperationOptimizer(
        model,
        block_info,
        alpha_provider,
        **settings,
    )


def test_abstract_contracts_cannot_be_instantiated():
    """두 추상 계약을 직접 생성할 수 없는지 검증한다. 입력은 없고 출력은 pytest 검증 결과이다. TypeError는 라이브러리 사용자가 구체 구현을 제공해야 함을 의미한다."""
    with pytest.raises(TypeError):
        buffer_module.ModelBlockInfo()
    with pytest.raises(TypeError):
        buffer_module.AlphaProvider()


def test_user_model_block_info_exposes_ordered_metadata(block_info):
    """사용자 구현체가 모델 생성과 ordered metadata를 제공하는지 검증한다. 입력은 FakeModelBlockInfo fixture이고 출력은 pytest 검증 결과이다. 이름과 그룹의 같은 위치는 동일한 block이어야 한다."""
    built_model = block_info.build_model(num_classes=2)
    names, groups = block_info.get_block_info(built_model)

    assert isinstance(built_model, TinyBlockModel)
    assert block_info.get_non_save_layers() == ["bias"]
    assert names == ["block_a.weight", "block_b.weight"]
    assert groups[0][0] is built_model.block_a.weight
    assert groups[1][0] is built_model.block_b.weight


def test_constructor_rejects_mismatched_names_and_groups(model, alpha_provider):
    """대표 이름과 파라미터 그룹의 개수 불일치를 거부하는지 검증한다. 입력은 모델과 alpha fixture이고 출력은 pytest 검증 결과이다. ValueError는 block 순서 불변식이 초기화 시점에 깨졌음을 의미한다."""
    with pytest.raises(ValueError, match="name|group|block"):
        build_optimizer(model, MismatchedModelBlockInfo(), alpha_provider)


def test_constructor_rejects_mismatched_alpha_count(model, block_info):
    """block 수와 alpha 수의 불일치를 거부하는지 검증한다. 입력은 모델과 block fixture이고 출력은 pytest 검증 결과이다. ValueError는 학습률 배수의 순서 대응이 불가능함을 의미한다."""
    with pytest.raises(ValueError, match="alpha|block"):
        build_optimizer(model, block_info, MismatchedAlphaProvider())


def test_constructor_rejects_reordered_names_and_groups(model, alpha_provider):
    """모델 순회와 다른 대표 이름 및 그룹 순서를 거부하는지 검증한다. 입력은 모델과 alpha fixture이고 출력은 pytest 검증 결과이다. ValueError는 네 ordered sequence가 동일한 순서를 가져야 함을 의미한다."""
    with pytest.raises(ValueError, match="order"):
        build_optimizer(model, ReorderedModelBlockInfo(), alpha_provider)


def test_constructor_builds_ordered_sgd_groups(model, block_info, alpha_provider):
    """생성자가 block 순서대로 SGD 그룹을 구성하는지 검증한다. 입력은 모델과 두 구현체 fixture이고 출력은 pytest 검증 결과이다. 각 그룹은 동일한 초기 learning rate와 momentum을 가져야 한다."""
    optimizer = build_optimizer(model, block_info, alpha_provider)
    groups = optimizer._optimizer.param_groups

    assert len(groups) == 2
    assert groups[0]["params"][0] is model.block_a.weight
    assert groups[1]["params"][0] is model.block_b.weight
    assert [group["lr"] for group in groups] == [0.1, 0.1]
    assert [group["momentum"] for group in groups] == [0.9, 0.9]


def test_step_lr_keeps_only_three_recent_snapshots(
    monkeypatch,
    model,
    block_info,
    alpha_provider,
):
    """epoch snapshot이 최근 세 개로 제한되는지 검증한다. 입력은 monkeypatch와 모델 및 두 구현체 fixture이고 출력은 pytest 검증 결과이다. 네 번의 호출 뒤 buffer에는 두 번째부터 네 번째 snapshot만 남아야 한다."""
    optimizer = build_optimizer(
        model,
        block_info,
        alpha_provider,
        epoch_offset=10,
    )
    snapshot_number = {"value": 0}

    def create_numbered_snapshot(parameter):
        """호출 순서가 담긴 가짜 subspace를 생성한다. 입력은 nn.Parameter이고 출력은 torch.Tensor이다. tensor 값은 오래된 snapshot이 정확히 제거됐는지 식별하는 번호를 의미한다."""
        snapshot_number["value"] += 1
        return torch.tensor([[float(snapshot_number["value"])]])

    monkeypatch.setattr(optimizer, "_create_subspace", create_numbered_snapshot)

    for epoch in range(4):
        optimizer.step_lr(epoch)

    assert list(optimizer._buffer) == ["block_a.weight", "block_b.weight"]
    assert all(isinstance(snapshots, deque) for snapshots in optimizer._buffer.values())
    assert all(snapshots.maxlen == 3 for snapshots in optimizer._buffer.values())
    assert all(len(snapshots) == 3 for snapshots in optimizer._buffer.values())
    assert [snapshot.item() for snapshot in optimizer._buffer["block_a.weight"]] == [3.0, 5.0, 7.0]
    assert [snapshot.item() for snapshot in optimizer._buffer["block_b.weight"]] == [4.0, 6.0, 8.0]


def test_step_lr_calls_internal_operations_in_epoch_order(
    monkeypatch,
    model,
    block_info,
    alpha_provider,
):
    """step_lr이 epoch 조건에 맞는 내부 연산 순서를 지키는지 검증한다. 입력은 monkeypatch와 세 fixture이고 출력은 pytest 검증 결과이다. 매 호출의 update와 기준점 및 갱신 epoch의 계산 순서가 보장되어야 한다."""
    optimizer = build_optimizer(model, block_info, alpha_provider)
    calls = []
    for snapshots in optimizer._buffer.values():
        snapshots.extend([object(), object(), object()])

    def record_update():
        """update 호출 사실을 기록한다. 입력은 없고 출력은 None이다. 기록값은 step_lr이 항상 update로 시작하는지 나타낸다."""
        calls.append("update")

    def record_set_basis():
        """기준점 설정 호출 사실을 기록한다. 입력은 없고 출력은 None이다. 기록값은 기준 epoch에서만 기준점이 설정되는지 나타낸다."""
        calls.append("set_basis")

    def record_calc_magnitude():
        """magnitude 계산 호출을 기록하고 두 값을 반환한다. 입력은 없고 출력은 float 원소를 갖는 list이다. 반환값은 후속 optimizer 갱신에 전달될 ordered magnitude이다."""
        calls.append("calc_magnitude")
        return [1.5, 0.5]

    def record_update_optimizer(magnitudes):
        """학습률 갱신 호출과 입력값을 기록한다. 입력은 float 원소를 갖는 list이고 출력은 None이다. 기록값은 계산 결과가 순서대로 전달되었는지 나타낸다."""
        calls.append(("update_optimizer", magnitudes))

    monkeypatch.setattr(optimizer, "_update", record_update)
    monkeypatch.setattr(optimizer, "_set_basis", record_set_basis)
    monkeypatch.setattr(optimizer, "_calc_magnitude", record_calc_magnitude)
    monkeypatch.setattr(optimizer, "_update_optimizer", record_update_optimizer)

    optimizer.step_lr(0)
    optimizer.step_lr(1)
    optimizer.step_lr(2)
    optimizer.step_lr(3)
    optimizer.step_lr(4)
    optimizer.step_lr(5)

    assert calls == [
        "update",
        "update",
        "update",
        "set_basis",
        "update",
        "calc_magnitude",
        ("update_optimizer", [1.5, 0.5]),
        "update",
        "update",
        "calc_magnitude",
        ("update_optimizer", [1.5, 0.5]),
    ]


def test_step_lr_raises_runtime_error_when_three_snapshots_are_unavailable(
    model,
    block_info,
    alpha_provider,
):
    """기준 epoch에 snapshot이 부족하면 명확히 실패하는지 검증한다. 입력은 모델과 두 구현체 fixture이고 출력은 pytest 검증 결과이다. RuntimeError는 세 snapshot 없이는 기준 magnitude를 계산할 수 없음을 의미한다."""
    optimizer = build_optimizer(
        model,
        block_info,
        alpha_provider,
        epoch_offset=1,
    )

    with pytest.raises(RuntimeError, match="3|three|snapshot|subspace"):
        optimizer.step_lr(0)


def test_update_optimizer_applies_base_lr_magnitude_and_alpha(
    model,
    block_info,
    alpha_provider,
):
    """block별 학습률이 세 계수의 곱으로 계산되는지 검증한다. 입력은 모델과 두 구현체 fixture이고 출력은 pytest 검증 결과이다. 각 결과는 base_lr과 ordered magnitude 및 alpha의 곱을 의미한다."""
    optimizer = build_optimizer(model, block_info, alpha_provider)

    optimizer._update_optimizer([1.5, 0.5])

    learning_rates = [group["lr"] for group in optimizer._optimizer.param_groups]
    assert learning_rates == pytest.approx([0.1 * 1.5 * 2.0, 0.1 * 0.5 * 3.0])


def test_step_updates_parameters_through_internal_sgd(
    model,
    block_info,
    alpha_provider,
):
    """step이 내부 SGD를 통해 실제 파라미터를 변경하는지 검증한다. 입력은 모델과 두 구현체 fixture이고 출력은 pytest 검증 결과이다. 변경된 weight는 이미 계산된 gradient가 SGD에 적용되었음을 의미한다."""
    optimizer = build_optimizer(
        model,
        block_info,
        alpha_provider,
        momentum=0.0,
    )
    before_a = model.block_a.weight.detach().clone()
    before_b = model.block_b.weight.detach().clone()
    model.block_a.weight.grad = torch.ones_like(model.block_a.weight)
    model.block_b.weight.grad = torch.ones_like(model.block_b.weight)

    optimizer.step()

    torch.testing.assert_close(model.block_a.weight, before_a - 0.1)
    torch.testing.assert_close(model.block_b.weight, before_b - 0.1)


@pytest.mark.parametrize("reshape_mode", ["in_dim", "out_dim", "ker_dim", "flatten"])
@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 2, 2)])
def test_cuda_subspace_matches_cpu(model, block_info, alpha_provider, reshape_mode, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is unavailable")
    pytest.importorskip("cupy")
    optimizer = build_optimizer(model, block_info, alpha_provider, reshape_mode=reshape_mode)
    generator = torch.Generator().manual_seed(42)
    cpu_snapshots, gpu_snapshots = deque(), deque()
    for _ in range(3):
        weight = torch.randn(shape, generator=generator, dtype=torch.float64)
        cpu = optimizer._create_subspace(nn.Parameter(weight))
        gpu = optimizer._create_subspace(nn.Parameter(weight.cuda()))
        assert gpu.is_cuda
        torch.testing.assert_close(cpu @ cpu.T, (gpu @ gpu.T).cpu(), atol=1e-8, rtol=1e-8)
        cpu_snapshots.append(cpu)
        gpu_snapshots.append(gpu)
    assert optimizer._measure_acceleration(gpu_snapshots) == pytest.approx(
        optimizer._measure_acceleration(cpu_snapshots), abs=1e-7
    )


def test_cpu_snapshot_is_independent(model, block_info, alpha_provider):
    optimizer = build_optimizer(model, block_info, alpha_provider, reshape_mode="flatten")
    snapshot = optimizer._create_subspace(model.block_a.weight)
    original = snapshot.clone()
    with torch.no_grad():
        model.block_a.weight.zero_()
    torch.testing.assert_close(snapshot, original)
