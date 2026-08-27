"""Single point where the app reads a secret.

Today the only backend is ``env`` (read ``os.environ``), which is exactly what the code
did inline before. The value of routing through here is the seam: at deploy time a
managed vault (AWS Secrets Manager, HashiCorp Vault, Doppler, SOPS-decrypted file, ...)
injects the secret and this module resolves it, with **no change to application code** —
only ``FACTORY_FLOOR_SECRETS_BACKEND`` changes. See ``docs/secrets.md``.

Note: this shadows the stdlib ``secrets`` module name *within this package only*.
Python 3 uses absolute imports, so ``import secrets`` elsewhere still gets the stdlib.
"""

import os

from factory_floor.config import get_settings

_DESIGN_ONLY_BACKENDS = {"aws", "vault", "doppler", "sops"}


def get_secret(name: str, default: str | None = None) -> str | None:
    """Resolve secret ``name`` via the configured backend.

    ``env``  -> ``os.getenv(name, default)`` (the current behaviour).
    others   -> raise ``NotImplementedError`` — the adapter seam is here, the client
                call is not wired until there is a real deployment target.
    """
    backend = get_settings().secrets_backend

    if backend == "env":
        return os.getenv(name, default)

    if backend in _DESIGN_ONLY_BACKENDS:
        raise NotImplementedError(
            f"secrets_backend={backend!r} is a design-only seam. Wire the real client "
            f"here (see docs/secrets.md) and return the resolved value for {name!r}."
        )

    raise ValueError(f"unknown FACTORY_FLOOR_SECRETS_BACKEND: {backend!r}")
