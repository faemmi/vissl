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
