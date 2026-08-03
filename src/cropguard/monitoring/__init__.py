"""Production monitoring: drift detection over prediction and confidence distributions.

Torch-free, like `cropguard.evaluation` — monitoring runs beside the serving container, and
that container has no torch in it.
"""
