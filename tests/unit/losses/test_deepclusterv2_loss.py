import pytest
import torch
import vissl.losses.deepclusterv2_loss as dcv2_loss


@pytest.mark.parametrize(
    ("tensor", "value", "expected"),
    [
        (
            torch.Tensor([-1, 0, 1]).int(),
            -2,
            False,
        ),
        (torch.Tensor([-1, 0, 1]).int(), -1, True),
        (
            torch.Tensor([0.0, 1.0, -1.0]).float(),
            -1.0,
            True,
        ),
    ],
)
def test_tensor_contains_value(tensor, value, expected):
    result = dcv2_loss._tensor_contains_value(tensor, value=value)

    assert result == expected


@pytest.mark.parametrize(
    ("tensor", "value", "expected"),
    [
        (torch.Tensor([-1, 0, 1]).int(), -2, torch.Tensor([]).int()),
        (torch.Tensor([-1, 0, 1]).int(), -1, torch.Tensor([0]).int()),
        (torch.Tensor([0.0, 1.0, -1.0]).float(), -1.0, torch.Tensor([2]).int()),
    ],
)
def test_tensor_contains_value_at(tensor, value, expected):
    result = dcv2_loss._tensor_contains_value_at(tensor, value=value)

    torch.testing.assert_equal(result, expected)
