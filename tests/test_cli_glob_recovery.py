"""Tests for the CLI's argv pre-processor and shell-glob recovery.

When a user runs the tool from a shell that eagerly expands globs
(cmd.exe on some configs, globstar-off bash with `**/...`), the pattern
`--include "**/*.md"` arrives in Python as a list of file paths instead
of one literal string. Click then sees those file paths as unexpected
extra arguments. The pre-processor repairs argv so the tool keeps
working, and warns the user so they can single-quote the glob next time.
"""

from __future__ import annotations

from pathlib import Path

from text_theme_analyzer.cli import _absorb_expanded_glob_args, _looks_like_path


# --- _looks_like_path ---

def test_looks_like_path_recognizes_paths() -> None:
    assert _looks_like_path("foo.md")
    assert _looks_like_path("a/b/c.md")
    assert _looks_like_path("a\\b\\c.md")
    assert _looks_like_path("./relative.md")
    assert _looks_like_path("C:/Users/me/note.markdown")
    assert _looks_like_path("queue.csv")


def test_looks_like_path_rejects_flags() -> None:
    assert not _looks_like_path("--include")
    assert not _looks_like_path("-o")
    assert not _looks_like_path("")


def test_looks_like_path_rejects_globs() -> None:
    # Globs with wildcards must NOT be absorbed as expanded paths.
    assert not _looks_like_path("**/*.md")
    assert not _looks_like_path("*.md")
    assert not _looks_like_path("foo?bar.md")
    assert not _looks_like_path("[abc].md")


def test_looks_like_path_rejects_bare_words() -> None:
    # A bare word with no separator and no extension: probably a flag value.
    assert not _looks_like_path("markdown")
    assert not _looks_like_path("json")


# --- _absorb_expanded_glob_args ---

def test_quoted_glob_is_unchanged() -> None:
    """A user who properly single-quoted their glob gets the literal
    pattern through to Click."""
    argv = ["analyze", "in", "--include", "**/*.md", "--include", "**/*.csv"]
    out = _absorb_expanded_glob_args(argv)
    assert out == argv


def test_shell_expanded_glob_gets_absorbed() -> None:
    """If the shell expanded `**/*.md` into file paths, those paths are
    folded into the include list (each as its own --include token) and
    Click won't see them as extras."""
    argv = [
        "analyze", "in",
        "--include", "**/*.md",
        "out/old.md",         # shell-expanded, looks like a path
        "src/foo.md",
        "--include", "**/*.csv",
    ]
    out = _absorb_expanded_glob_args(argv)
    # The expanded paths are rewritten as their own --include tokens.
    assert "**/*.md" in out
    assert "out/old.md" in out
    assert "src/foo.md" in out
    assert "**/*.csv" in out
    # Each expanded path has its own --include in front of it.
    assert out[out.index("out/old.md") - 1] == "--include"
    assert out[out.index("src/foo.md") - 1] == "--include"


def test_no_include_flag_means_nothing_to_absorb() -> None:
    argv = ["analyze", "in", "-o", "markdown,html"]
    assert _absorb_expanded_glob_args(argv) == argv


def test_include_followed_immediately_by_next_flag() -> None:
    """`--include "**/*.md" --exclude "**/archive/**"` -> no expansion."""
    argv = ["analyze", "in", "--include", "**/*.md", "--exclude", "**/archive/**"]
    assert _absorb_expanded_glob_args(argv) == argv


def test_only_path_like_tokens_get_absorbed() -> None:
    """Non-path-looking tokens stop the absorption (they're a new option)."""
    argv = [
        "analyze", "in",
        "--include", "**/*.md",
        "real/path.md",       # absorb
        "another.md",         # absorb
        "not-a-path",         # stop (no separator, no extension)
        "--include", "**/*.csv",
    ]
    out = _absorb_expanded_glob_args(argv)
    assert "real/path.md" in out
    assert "another.md" in out
    # 'not-a-path' is preserved as a separate arg (Click will probably reject it
    # with a different error, but the pre-processor doesn't have to know that).
    assert "not-a-path" in out


def test_shell_ate_the_whole_include_flag() -> None:
    """Worst case: the shell expanded `--include "**/*.md"` to N paths
    AND ate the `--include` flag itself. The files arrive at the start
    of argv (after INPUT_PATH). Phase 2 of the repair wraps each in
    its own --include token."""
    argv = [
        "analyze", "in",
        "out/old.md",         # shell ate --include, expanded to these
        "src/foo.md",
        "--include", "**/*.csv",
        "-o", "markdown",
    ]
    out = _absorb_expanded_glob_args(argv)
    # Each leading path is wrapped in its own --include.
    assert out[2] == "--include"
    assert out[3] == "out/old.md"
    assert out[4] == "--include"
    assert out[5] == "src/foo.md"
    assert out[6] == "--include"
    assert out[7] == "**/*.csv"


def test_diff_command_is_not_repaired() -> None:
    """The `diff` subcommand doesn't take --include. Path-like tokens
    between `diff OLD NEW` are the positional args and must be left alone."""
    argv = ["diff", "snapshot1", "snapshot2", "--output-dir", "out/"]
    out = _absorb_expanded_glob_args(argv)
    assert out == argv


def test_empty_argv() -> None:
    assert _absorb_expanded_glob_args([]) == []


def test_absorbed_paths_survive_into_click() -> None:
    """End-to-end: with shell-expanded globs, the CLI shouldn't crash
    with 'unexpected extra arguments'."""
    from click.testing import CliRunner
    from text_theme_analyzer.cli import main

    runner = CliRunner()
    # Create a temp corpus with two .md files.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "a.md").write_text("# a\n\nbody a", encoding="utf-8")
        (td_path / "b.md").write_text("# b\n\nbody b", encoding="utf-8")
        (td_path / "queue.csv").write_text("id,draft\n1,hi\n", encoding="utf-8")

        # Simulate the shell-expanded argv: --include "**/*.md" was expanded
        # to two file paths that the user accidentally pasted after the flag.
        result = runner.invoke(
            main,
            [
                "analyze", str(td_path),
                "--include", "**/*.md",
                str(td_path / "a.md"),
                str(td_path / "b.md"),
                "--include", "**/*.csv",
                "-o", "json",
                "--no-llm",
                "--no-history",
                "-q",
            ],
        )
        # Should NOT have raised the "Got unexpected extra arguments" error.
        assert "unexpected extra argument" not in result.output.lower(), result.output
        # And it should have found the notes.
        assert result.exit_code == 0, f"exit {result.exit_code}: {result.output}"
