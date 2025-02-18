import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import DataClassJsonMixin  # optional if you want to .to_json()
import contextlib
import io
import humanize


logger = logging.getLogger("aide")


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """
    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


class Interpreter:
    def __init__(
        self,
        working_dir: str | Path,
        timeout: int = 300,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
    ):
        """
        In-process Interpreter that runs code in the current process.

        Args:
            working_dir (Path | str): Directory in which to operate.
            timeout (int, optional): (Not used in-process; no separate process.)
            format_tb_ipython (bool, optional): If you want IPython-style tracebacks.
            agent_file_name (str, optional): If you want to store code to disk.
        """
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # The "timeout" is not enforced in-process by default.
        # If you want, you can do advanced approaches like signal.alarm().
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name

        # Keep a global scope so multiple runs can share variables,
        # or re-init in run() if reset_session=True
        self._global_scope = {}

    def cleanup_session(self):
        """
        Since we're in-process, there's no separate process to terminate.
        Clean up temporary files here if desired.
        """
        pass

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """
        Execute the provided Python code in this process, capturing stdout & stderr.

        Args:
            code (str): Python code to execute.
            reset_session (bool): If True, re-initialize the global scope each time.

        Returns:
            ExecutionResult: Object containing stdout/stderr, exec time, exception info, etc.
        """
        logger.info(f"Running code in-process (reset_session={reset_session}).")

        if reset_session:
            self._global_scope = {}

        # Optionally write code to a file in working_dir
        runfile_path = self.working_dir / self.agent_file_name
        with open(runfile_path, "w", encoding="utf-8") as f:
            f.write(code)

        start_time = time.time()
        out_buf = io.StringIO()

        exc_type = None
        exc_info = None
        exc_stack = None
        lines_out = []

        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(out_buf):
            # Temporarily switch current working directory so user code writes 
            # local files to self.working_dir
            old_cwd = os.getcwd()
            try:
                os.chdir(str(self.working_dir))

                try:
                    compiled_code = compile(code, filename=str(runfile_path), mode="exec")
                    exec(compiled_code, self._global_scope)
                except BaseException as e:
                    # Gather traceback info
                    tb_str, e_cls_name, e_info, e_stack = self._format_exception(e, runfile_path)
                    lines_out.append(tb_str)
                    exc_type = e_cls_name
                    exc_info = e_info
                    exc_stack = e_stack

            finally:
                os.chdir(old_cwd)

        out_val = out_buf.getvalue()
        if out_val.strip():
            lines_out.extend(out_val.splitlines())

        exec_time = time.time() - start_time
        lines_out.append(f"Execution time: {humanize.naturaldelta(exec_time)}.")

        return ExecutionResult(
            term_out=lines_out,
            exec_time=exec_time,
            exc_type=exc_type,
            exc_info=exc_info,
            exc_stack=exc_stack,
        )

    def _format_exception(self, e: BaseException, runfile_path: Path):
        """
        Helper: format an exception stack trace.
        If you want IPython style, set self.format_tb_ipython = True.
        """
        if self.format_tb_ipython:
            import IPython.core.ultratb
            tb = IPython.core.ultratb.VerboseTB(color_scheme="NoColor")
            tb_str = tb.text(*sys.exc_info())
        else:
            tb_lines = traceback.format_exception(e.__class__, e, e.__traceback__)
            tb_str = "".join(tb_lines)

        # Optionally replace the full path with just filename, for cleanliness
        tb_str = tb_str.replace(str(runfile_path), self.agent_file_name)

        e_cls_name = e.__class__.__name__
        exc_info = {}
        if hasattr(e, "args"):
            exc_info["args"] = [str(arg) for arg in e.args]

        tb = traceback.extract_tb(e.__traceback__)
        exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

        return tb_str, e_cls_name, exc_info, exc_stack
