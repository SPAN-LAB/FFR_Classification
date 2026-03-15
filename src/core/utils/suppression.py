import os
import sys

def silence_stderr(func, *args, **kwargs):
    """Wraps a function to suppress its stderr."""
    with open(os.devnull, 'w') as null:
        # Save original stderr
        old_stderr = os.dup(sys.stderr.fileno())
        # Replace stderr with null
        os.dup2(null.fileno(), sys.stderr.fileno())
        try:
            return func(*args, **kwargs)
        finally:
            # Restore original stderr
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stderr)