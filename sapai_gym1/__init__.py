"""Super Auto Pets Gym package.

Keep the heavyweight Gym/Torch environment lazy so lightweight live-capture
clients can import ``sapai_gym1.deployment`` without installing the complete
training stack.
"""

__all__ = ["SuperAutoPetsEnv"]


def __getattr__(name):
    if name == "SuperAutoPetsEnv":
        import importlib

        module = importlib.import_module(".SuperAutoPetsEnv", __name__)
        environment_class = module.SuperAutoPetsEnv
        # Importing a child module sets ``package.SuperAutoPetsEnv`` to that
        # module. Replace it after import so the historical public API keeps
        # returning the environment class.
        globals()[name] = environment_class
        return environment_class
    raise AttributeError(name)
