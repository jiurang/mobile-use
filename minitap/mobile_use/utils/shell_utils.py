import subprocess

# Without a timeout a hung `adb`/`ios-deploy` blocks its caller forever. On the
# agent's shared event loop that is an unrecoverable UI freeze, not just a slow
# call, so the bound is deliberate rather than decorative.
DEFAULT_SHELL_TIMEOUT_SECONDS = 30


def run_shell_command_on_host(
    command: str,
    timeout: float = DEFAULT_SHELL_TIMEOUT_SECONDS,
) -> str:
    """Helper to run a shell command on the host and return the output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Command '{command}' timed out after {timeout}s.") from e
    except subprocess.CalledProcessError as e:
        # Log the error and stderr for better debugging
        error_message = (
            f"Command '{command}' failed with exit code {e.returncode}. Stderr: {e.stderr.strip()}"
        )
        raise RuntimeError(error_message) from e
