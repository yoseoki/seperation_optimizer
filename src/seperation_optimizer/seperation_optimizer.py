import math
from abc import ABC, abstractmethod
from collections import deque

import torch


class ModelBlockInfo(ABC):
    """모델 생성과 block metadata의 사용자 계약을 정의한다."""

    @abstractmethod
    def build_model(self, num_classes):
        """학습에 사용할 모델을 생성한다. 입력은 int이고 출력은 Module이다. num_classes는 모델이 예측할 클래스 수를 의미한다."""
        raise NotImplementedError

    @abstractmethod
    def get_non_save_layers(self):
        """저장과 분석에서 제외할 레이어 이름을 반환한다. 입력은 없고 출력은 str 원소를 갖는 list이다. 각 문자열은 대표 파라미터 선택에서 제외할 이름 조각을 의미한다."""
        raise NotImplementedError

    @abstractmethod
    def get_block_info(self, model):
        """대표 레이어 이름과 파라미터 그룹을 순서대로 반환한다. 입력은 Module이고 출력은 두 list를 담은 tuple이다. 두 list의 같은 위치는 동일한 모델 block을 의미한다."""
        raise NotImplementedError


class AlphaProvider(ABC):
    """block별 고정 alpha를 제공하는 사용자 계약을 정의한다."""

    @abstractmethod
    def get_alphas(self):
        """block 순서에 대응하는 alpha 목록을 반환한다. 입력은 없고 출력은 수치 원소를 갖는 list이다. 각 값은 해당 block의 magnitude에 곱할 고정 배수를 의미한다."""
        raise NotImplementedError


class SeperationOptimizer:
    """부분공간 기반 학습률과 내부 SGD를 함께 관리한다."""

    _EPSILON = 1e-12

    def __init__(
        self,
        model,
        model_block_info,
        alpha_provider,
        base_lr,
        momentum=0.9,
        weight_decay=0.0,
        epoch_offset=5,
        epoch_interval=5,
        reshape_mode="in_dim",
    ):
        """주입된 block 정보로 내부 SGD와 부분공간 상태를 초기화한다. 입력은 모델과 두 계약 구현체 및 optimizer 설정이고 출력은 None이다. 초기화된 상태는 대표 레이어와 파라미터 그룹 및 alpha의 동일한 순서를 보장한다."""
        if not isinstance(model_block_info, ModelBlockInfo):
            raise TypeError("model_block_info must implement ModelBlockInfo")
        if not isinstance(alpha_provider, AlphaProvider):
            raise TypeError("alpha_provider must implement AlphaProvider")
        if base_lr < 0:
            raise ValueError("base_lr must be non-negative")
        if momentum < 0:
            raise ValueError("momentum must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if epoch_offset < 1:
            raise ValueError("epoch_offset must be at least 1")
        if epoch_interval < 1:
            raise ValueError("epoch_interval must be at least 1")
        if reshape_mode not in {"in_dim", "out_dim", "ker_dim", "flatten"}:
            raise ValueError("reshape_mode must be in_dim, out_dim, ker_dim, or flatten")

        self.model = model
        self.model_block_info = model_block_info
        self.alpha_provider = alpha_provider
        self.base_lr = float(base_lr)
        self.epoch_offset = int(epoch_offset)
        self.epoch_interval = int(epoch_interval)
        self.reshape_mode = reshape_mode

        names, parameter_groups = self.model_block_info.get_block_info(self.model)
        self._names = list(names)
        self._parameter_groups = [list(group) for group in parameter_groups]
        self._alphas = list(self.alpha_provider.get_alphas())

        if not self._names:
            raise ValueError("at least one block name and parameter group are required")
        if len(self._names) != len(self._parameter_groups):
            raise ValueError("block name and parameter group counts must match")
        if len(self._alphas) != len(self._names):
            raise ValueError("alpha and block counts must match")
        if len(set(self._names)) != len(self._names):
            raise ValueError("block names must be unique")
        if not all(isinstance(name, str) and name for name in self._names):
            raise ValueError("block names must be non-empty strings")
        if not all(group for group in self._parameter_groups):
            raise ValueError("parameter groups must not be empty")

        for alpha in self._alphas:
            if not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
                raise ValueError("alpha values must be finite numbers")
            if alpha < 0:
                raise ValueError("alpha values must be non-negative")

        self._representative_parameters = self._select_representative_parameters()
        optimizer_groups = []
        for group in self._parameter_groups:
            optimizer_groups.append(
                {
                    "params": group,
                    "lr": self.base_lr,
                    "momentum": float(momentum),
                }
            )

        self._optimizer = torch.optim.SGD(
            optimizer_groups,
            lr=self.base_lr,
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
        self._buffer = {name: deque(maxlen=3) for name in self._names}
        self._base_magnitudes = {name: None for name in self._names}

    def _select_representative_parameters(self):
        """ordered name마다 하나의 대표 파라미터를 선택하고 순서를 검증한다. 입력은 없고 출력은 name과 Parameter 쌍을 갖는 list이다. 반환 순서는 block과 SGD parameter group 및 alpha 순서와 일치한다."""
        excluded_names = list(self.model_block_info.get_non_save_layers())
        named_parameters = list(self.model.named_parameters())
        parameter_positions = {id(parameter): index for index, (_, parameter) in enumerate(named_parameters)}
        representatives = []
        previous_position = -1

        for block_index, representative_name in enumerate(self._names):
            matches = []
            for parameter_name, parameter in named_parameters:
                is_representative = (
                    parameter_name == representative_name
                    or parameter_name.startswith(representative_name + ".")
                )
                is_excluded = any(excluded in parameter_name for excluded in excluded_names)
                if is_representative and not is_excluded:
                    matches.append((parameter_name, parameter))

            if len(matches) != 1:
                raise ValueError(
                    "each block name must select exactly one representative parameter: "
                    + representative_name
                )

            parameter_name, parameter = matches[0]
            position = parameter_positions[id(parameter)]
            if position <= previous_position:
                raise ValueError("representative parameter order must match block name order")
            if not any(parameter is candidate for candidate in self._parameter_groups[block_index]):
                raise ValueError("representative parameter must belong to its parameter group")

            representatives.append((parameter_name, parameter))
            previous_position = position

        return representatives

    def _orthonormalize(self, matrix):
        """행렬의 유효한 열 공간에 대한 직교기저를 계산한다. 입력과 출력은 2차원 Tensor이다. 반환 열들은 수치적으로 유효한 원래 행렬의 부분공간을 의미한다."""
        if matrix.ndim != 2:
            raise ValueError("subspace input must be a two-dimensional tensor")
        if matrix.numel() == 0:
            return torch.zeros((matrix.shape[0], 1), dtype=matrix.dtype)
        if not torch.isfinite(matrix).all():
            raise RuntimeError("representative weight contains NaN or Inf")

        left_vectors, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        if singular_values.numel() == 0:
            return torch.zeros((matrix.shape[0], 1), dtype=matrix.dtype)

        largest_value = singular_values.max().item()
        if largest_value <= self._EPSILON:
            return torch.zeros((matrix.shape[0], 1), dtype=matrix.dtype)

        tolerance = self._EPSILON * max(matrix.shape) * largest_value
        rank = int((singular_values > tolerance).sum().item())
        if rank == 0:
            return torch.zeros((matrix.shape[0], 1), dtype=matrix.dtype)
        return left_vectors[:, :rank].contiguous()

    def _create_subspace(self, parameter):
        """대표 weight와 같은 device에서 직교 부분공간을 생성한다. CUDA에서는 CuPy를 사용하며 출력은 2차원 Tensor이다. 반환값은 epoch 시점의 대표 레이어 weight 열 공간을 의미한다."""
        if parameter.is_cuda:
            return self._create_cuda_subspace(parameter)
        weight = parameter.detach().to(device="cpu", dtype=torch.float64).clone()

        if self.reshape_mode == "flatten" or weight.ndim < 2:
            vector = weight.reshape(-1, 1)
            norm = torch.linalg.vector_norm(vector)
            if norm.item() <= self._EPSILON:
                basis = torch.zeros_like(vector)
            else:
                basis = vector / norm
        elif weight.ndim == 4:
            out_channels, in_channels, height, width = weight.shape
            kernel_size = height * width
            if self.reshape_mode == "in_dim":
                data = weight.swapaxes(0, 1).reshape(in_channels, out_channels * kernel_size).T
            elif self.reshape_mode == "out_dim":
                data = weight.reshape(out_channels, in_channels * kernel_size).T
            elif self.reshape_mode == "ker_dim":
                data = weight.reshape(out_channels * in_channels, kernel_size).T
            basis = self._orthonormalize(data)
        else:
            out_dimensions = weight.shape[0]
            in_dimensions = math.prod(weight.shape[1:])
            data = weight.reshape(out_dimensions, in_dimensions)
            if in_dimensions >= out_dimensions:
                data = data.T
            basis = self._orthonormalize(data)

        if not torch.isfinite(basis).all():
            raise RuntimeError("subspace calculation produced NaN or Inf")
        return basis

    @staticmethod
    def _cupy():
        """CUDA 경로에서만 선택적 CuPy 의존성을 불러온다."""
        try:
            import cupy
        except ImportError as error:
            raise RuntimeError(
                "CUDA subspace calculations require CuPy. Install the cuda12 or "
                "cuda13 extra matching your PyTorch CUDA build; see README.md."
            ) from error
        return cupy

    def _cupy_orthonormalize(self, matrix, cp):
        """GPU 행렬의 유효 열 공간을 CuPy SVD로 계산한다."""
        if not bool(cp.isfinite(matrix).all().item()):
            raise RuntimeError("representative weight contains NaN or Inf")
        if matrix.size == 0:
            return cp.zeros((matrix.shape[0], 1), dtype=matrix.dtype)
        with cp.errstate(linalg="raise"):
            left, values, _ = cp.linalg.svd(matrix, full_matrices=False)
        largest = values.max().item() if values.size else 0.0
        if largest <= self._EPSILON:
            return cp.zeros((matrix.shape[0], 1), dtype=matrix.dtype)
        rank = int((values > self._EPSILON * max(matrix.shape) * largest).sum().item())
        if rank == 0:
            return cp.zeros((matrix.shape[0], 1), dtype=matrix.dtype)
        return cp.ascontiguousarray(left[:, :rank])

    def _create_cuda_subspace(self, parameter):
        """DLPack으로 GPU weight를 공유하고 독립적인 CuPy snapshot을 생성한다."""
        cp = self._cupy()
        with cp.cuda.Device(parameter.device.index):
            weight = cp.from_dlpack(parameter.detach().to(dtype=torch.float64)).copy()
            if not bool(cp.isfinite(weight).all().item()):
                raise RuntimeError("representative weight contains NaN or Inf")
            if self.reshape_mode == "flatten" or weight.ndim < 2:
                vector = weight.reshape(-1, 1)
                norm = cp.linalg.norm(vector).item()
                basis = cp.zeros_like(vector) if norm <= self._EPSILON else vector / norm
            else:
                if weight.ndim == 4:
                    out_channels, in_channels, height, width = weight.shape
                    kernel_size = height * width
                    if self.reshape_mode == "in_dim":
                        data = weight.swapaxes(0, 1).reshape(in_channels, -1).T
                    elif self.reshape_mode == "out_dim":
                        data = weight.reshape(out_channels, -1).T
                    else:
                        data = weight.reshape(out_channels * in_channels, kernel_size).T
                else:
                    data = weight.reshape(weight.shape[0], math.prod(weight.shape[1:]))
                    if data.shape[1] >= data.shape[0]:
                        data = data.T
                basis = self._cupy_orthonormalize(data, cp)
            if not bool(cp.isfinite(basis).all().item()):
                raise RuntimeError("subspace calculation produced NaN or Inf")
            return torch.from_dlpack(basis)

    def _measure_cuda_acceleration(self, snapshots):
        """snapshot의 투영, SVD, 가속도 계산을 GPU에서 수행한다."""
        cp = self._cupy()
        with cp.cuda.Device(snapshots[0].device.index):
            previous, current, following = [cp.from_dlpack(basis) for basis in snapshots]
            if cp.linalg.norm(current).item() <= self._EPSILON:
                return 0.0
            span = self._cupy_orthonormalize(cp.concatenate((previous, following), axis=1), cp)
            if cp.linalg.norm(span).item() <= self._EPSILON:
                return 0.0
            projected = self._cupy_orthonormalize(span @ (span.T @ current), cp)
            if cp.linalg.norm(projected).item() <= self._EPSILON:
                return float(2 * current.shape[1])
            with cp.errstate(linalg="raise"):
                values = cp.linalg.svd(projected.T @ current, compute_uv=False)
            magnitude = 2.0 * (current.shape[1] - cp.clip(values, 0.0, 1.0).sum().item())
            if not math.isfinite(magnitude):
                raise RuntimeError("acceleration magnitude calculation produced NaN or Inf")
            return max(0.0, magnitude)

    def _require_three_snapshots(self):
        """모든 대표 레이어에 최근 snapshot 세 개가 있는지 검증한다. 입력은 없고 출력은 None이다. RuntimeError는 기준값이나 magnitude 계산에 필요한 이력이 부족함을 의미한다."""
        missing_names = [name for name in self._names if len(self._buffer[name]) < 3]
        if missing_names:
            raise RuntimeError(
                "3 subspace snapshots are required for every block: "
                + ", ".join(missing_names)
            )

    def _measure_acceleration(self, snapshots):
        """세 subspace에서 직교 방향의 가속도 크기를 계산한다. 입력은 Tensor 세 개를 담은 deque이고 출력은 float이다. 결과는 가운데 subspace가 앞뒤 subspace의 결합 공간을 벗어난 정도를 의미한다."""
        basis_previous, basis_current, basis_next = snapshots
        if basis_previous.shape[0] != basis_current.shape[0] or basis_current.shape[0] != basis_next.shape[0]:
            raise RuntimeError("subspace dimensions must remain constant across snapshots")
        if any(basis.device != basis_current.device for basis in snapshots):
            raise RuntimeError("snapshot devices must remain constant")
        if basis_current.is_cuda:
            return self._measure_cuda_acceleration(snapshots)
        if torch.linalg.vector_norm(basis_current).item() <= self._EPSILON:
            return 0.0

        span = self._orthonormalize(torch.cat((basis_previous, basis_next), dim=1))
        if torch.linalg.vector_norm(span).item() <= self._EPSILON:
            return 0.0

        projected = span @ (span.T @ basis_current)
        projected_basis = self._orthonormalize(projected)
        if torch.linalg.vector_norm(projected_basis).item() <= self._EPSILON:
            return float(2 * basis_current.shape[1])

        singular_values = torch.linalg.svdvals(projected_basis.T @ basis_current)
        singular_values = singular_values.clamp(min=0.0, max=1.0)
        magnitude = 2.0 * (basis_current.shape[1] - singular_values.sum().item())
        if not math.isfinite(magnitude):
            raise RuntimeError("acceleration magnitude calculation produced NaN or Inf")
        return max(0.0, magnitude)

    def _update(self):
        """대표 레이어의 현재 weight로 subspace snapshot을 추가한다. 입력은 없고 출력은 None이다. 각 block의 buffer에는 계산에 필요한 최근 세 snapshot만 유지된다."""
        for block_name, (_, parameter) in zip(self._names, self._representative_parameters):
            self._buffer[block_name].append(self._create_subspace(parameter))

    def _calc_magnitude(self):
        """최근 세 subspace를 기준값과 비교한 block별 계수를 계산한다. 입력은 없고 출력은 float 원소를 갖는 ordered list이다. 각 값은 현재 acceleration magnitude를 초기 기준값으로 나눈 학습률 배수이다."""
        self._require_three_snapshots()
        magnitudes = []

        for block_name in self._names:
            base_magnitude = self._base_magnitudes[block_name]
            if base_magnitude is None:
                raise RuntimeError("base magnitude must be set before calculating magnitudes")

            current_magnitude = self._measure_acceleration(self._buffer[block_name])
            if base_magnitude <= self._EPSILON:
                ratio = 1.0
            else:
                ratio = current_magnitude / base_magnitude
            if not math.isfinite(ratio):
                ratio = 1.0
            magnitudes.append(float(ratio))

        return magnitudes

    def _set_basis(self):
        """초기 세 subspace로 block별 acceleration 기준값을 설정한다. 입력은 없고 출력은 None이다. 저장된 값은 이후 magnitude를 상대적인 학습률 배수로 변환하는 기준이다."""
        self._require_three_snapshots()
        for block_name in self._names:
            self._base_magnitudes[block_name] = abs(
                self._measure_acceleration(self._buffer[block_name])
            )

    def _update_optimizer(self, magnitudes):
        """magnitude와 alpha를 결합해 내부 SGD의 학습률을 갱신한다. 입력은 block 순서의 수치 list이고 출력은 None이다. 각 학습률은 base_lr과 magnitude 및 alpha의 곱을 의미한다."""
        if len(magnitudes) != len(self._names):
            raise ValueError("magnitude and block counts must match")

        for parameter_group, magnitude, alpha in zip(
            self._optimizer.param_groups,
            magnitudes,
            self._alphas,
        ):
            if not isinstance(magnitude, (int, float)) or not math.isfinite(magnitude):
                raise ValueError("magnitude values must be finite numbers")
            if magnitude < 0:
                raise ValueError("magnitude values must be non-negative")
            parameter_group["lr"] = self.base_lr * float(magnitude) * float(alpha)

    def step_lr(self, epoch_num):
        """epoch 시작 시 snapshot과 조건부 학습률 갱신을 수행한다. 입력은 0부터 시작하는 int이고 출력은 None이다. 기준 epoch에는 기준값을 설정하고 이후 interval epoch에는 내부 SGD 학습률을 갱신한다."""
        if not isinstance(epoch_num, int) or epoch_num < 0:
            raise ValueError("epoch_num must be a non-negative integer")

        self._update()
        if epoch_num == self.epoch_offset - 1:
            self._set_basis()
        elif (
            epoch_num >= self.epoch_offset
            and (epoch_num - self.epoch_offset) % self.epoch_interval == 0
        ):
            magnitudes = self._calc_magnitude()
            self._update_optimizer(magnitudes)

    def zero_grad(self, set_to_none=True):
        """내부 SGD가 관리하는 gradient를 초기화한다. 입력은 bool이고 출력은 None이다. set_to_none은 gradient를 0 tensor 대신 None으로 설정할지를 의미한다."""
        return self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """이미 계산된 gradient로 내부 SGD의 parameter 갱신을 수행한다. 입력은 선택적 callable이고 출력은 내부 SGD step의 반환값이다. 이 메소드는 forward나 loss 계산 및 backward를 수행하지 않는다."""
        return self._optimizer.step(closure=closure)
