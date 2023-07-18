import random

import pytest
import torch
import vissl.plotting.deepclusterv2.assignments as _assignments


def test_plot_abundance(tmp_path):
    # Assignments for 2 samples
    assignments = torch.Tensor(
        [-1, 0, 1, 2],
    )

    expected = [
        tmp_path / "abundance-total.pdf",
        tmp_path / "abundance-relative.pdf",
    ]

    _assignments.plot_abundance(
        assignments=assignments,
        name="abundance",
        output_dir=tmp_path.as_posix(),
    )

    assert all(plot.exists() for plot in expected)


def test_plot_appearance_per_week(tmp_path):
    assignments = torch.Tensor(
        [random.randint(-1, 30) for _ in range(365 * 2)],
    )

    expected = tmp_path / "appearance-per-week.pdf"

    _assignments.plot_appearance_per_week(
        assignments=assignments,
        name="appearance-per-week",
        output_dir=tmp_path.as_posix(),
    )

    assert expected.exists()


def test_reshape_assignments_to_weeks():
    assignments = torch.Tensor(list(range(1, 15)))

    expected = torch.Tensor(
        [
            [1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14],
        ]
    )

    result = _assignments._reshape_assignments_to_weeks(assignments)

    torch.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ("assignments", "expected"),
    [
        (torch.Tensor(list(range(1, 10))), 7),
        (torch.Tensor(list(range(1, 366))), 364),
    ],
)
def test_find_largest_dividable_by_7(assignments, expected):
    result = _assignments._find_largest_dividable_by_7(assignments)

    assert result == expected
