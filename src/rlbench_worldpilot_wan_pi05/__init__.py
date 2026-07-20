"""WorldPilot-style WAN latent steering utilities for RLBench pi0.5."""

__all__ = ["WanFutureVideoFuser"]


def __getattr__(name: str):
    # Keep the CPU-only RLBench RPC client importable in an environment that
    # does not install torch. Model processes still get the same public symbol.
    if name == "WanFutureVideoFuser":
        from .fusion import WanFutureVideoFuser

        return WanFutureVideoFuser
    raise AttributeError(name)
