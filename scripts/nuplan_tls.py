"""Verified TLS-context construction shared by the nuPlan range clients."""

from __future__ import annotations

import ssl
from pathlib import Path


class NuPlanTlsError(RuntimeError):
    """Raised when no usable certificate authority bundle is available."""


def verified_https_context() -> ssl.SSLContext:
    """Return a hostname-verifying context with at least one trusted CA.

    Some copied virtual environments retain a stale OpenSSL default path. In
    that case, use the environment's ``certifi`` bundle if it is installed.
    A TLS connection is never made with verification disabled.
    """

    default_error: BaseException | None = None
    try:
        context = ssl.create_default_context()
        if context.get_ca_certs():
            return context
    except (OSError, ssl.SSLError) as exc:
        default_error = exc

    try:
        import certifi

        ca_bundle = Path(certifi.where())
        if not ca_bundle.is_file():
            raise NuPlanTlsError(f"certifi returned a missing CA bundle: {ca_bundle}")
        context = ssl.create_default_context(cafile=str(ca_bundle))
        if context.get_ca_certs():
            return context
    except ImportError as exc:
        detail = "certifi is not installed"
        if default_error is not None:
            detail += f"; system trust-store error: {default_error}"
        raise NuPlanTlsError(f"No usable verified HTTPS trust store: {detail}") from exc
    except (OSError, ssl.SSLError) as exc:
        detail = f"certifi CA bundle could not be loaded: {exc}"
        if default_error is not None:
            detail += f"; system trust-store error: {default_error}"
        raise NuPlanTlsError(f"No usable verified HTTPS trust store: {detail}") from exc

    raise NuPlanTlsError("No usable verified HTTPS trust store: both system and certifi bundles are empty")
