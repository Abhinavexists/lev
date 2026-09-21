"""Benchmark harness for Jev (TypeSafe System One) against an LLM baseline.

Import the submodules directly -- `from levbench import runner, metrics`. The
package deliberately binds nothing at import time so that `levbench pricing`
does not drag in an HTTP client.
"""
