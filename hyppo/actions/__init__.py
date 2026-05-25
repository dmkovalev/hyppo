"""hyppo action registry — typed operations callable by agents over MCP.

Importing this package eagerly imports every action module so that the
``@action`` decorators populate ``ACTION_REGISTRY``.
"""
# Modules are added incrementally by tasks T2-T8. Keep imports alphabetical.
