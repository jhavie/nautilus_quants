"""Live module exceptions."""


class LiveError(Exception):
    """Base exception for the live module."""


class LiveConfigError(LiveError):
    """YAML config parsing/validation errors."""
