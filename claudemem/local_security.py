"""Network boundary checks for the single-user local HTTP service."""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit


def is_loopback_host(host: str | None) -> bool:
    value = (host or "").strip().strip("[]").rstrip(".").lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def is_safe_origin(origin: str | None) -> bool:
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and is_loopback_host(parsed.hostname)
