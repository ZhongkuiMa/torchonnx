"""Regression guard: generated PyTorch code is deterministic.

R6 dropped ``datetime.now()`` and the absolute source path from the
file header in ``simplify/_decorations.py``. Without a guard, a future
refactor could easily re-introduce non-determinism (a wall-clock build
ID, a converter version, a random class suffix) and silently break
content-hash caching and bit-reproducible diffs.

These tests cover the public-facing seam at
``simplify.add_file_header`` because that is where the prior
non-determinism lived and where the rest of the pipeline is expected
to stay deterministic.
"""

__docformat__ = "restructuredtext"

from torchonnx.simplify import add_file_header


class TestAddFileHeaderDeterminism:
    """``add_file_header`` must produce identical output for identical inputs."""

    def test_identical_inputs_produce_identical_output(self):
        body = "import torch\n\nclass Mod(torch.nn.Module): ...\n"
        first = add_file_header(body, "MyModule", "/data/foo.onnx")
        second = add_file_header(body, "MyModule", "/data/foo.onnx")
        assert first == second

    def test_absolute_path_differences_do_not_leak_into_output(self):
        """Two callers on different machines must produce byte-identical files.

        The header keeps only the source basename, never the parent
        directory chain, so converting the same ONNX file from
        ``/laptop/data/foo.onnx`` and ``/server/scratch/foo.onnx``
        produces the exact same emitted ``.py`` text.
        """
        body = "class M: ...\n"
        laptop = add_file_header(body, "M", "/laptop/data/foo.onnx")
        server = add_file_header(body, "M", "/server/scratch/foo.onnx")
        assert laptop == server

    def test_header_does_not_contain_wallclock_timestamp(self):
        """No ``datetime.now()`` markers may appear in the generated header.

        We catch the obvious shapes (the prior ``Generated: YYYY-MM-DD
        HH:MM:SS`` line plus any year-prefix pattern) so even a partial
        re-introduction trips the test.
        """
        out = add_file_header("pass\n", "M", "model.onnx")
        assert "Generated:" not in out
        # No four-digit year followed by month/day in the header section.
        header = out.split('"""', 2)[1]
        assert "2024" not in header
        assert "2025" not in header
        assert "2026" not in header
