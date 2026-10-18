# Lazy exports so importing e.g. engine.weights doesn't pull in torch on the UI thread.
def __getattr__(name):
    if name in ("Dreamer", "DreamSettings", "Preview"):
        from . import dreamer
        return getattr(dreamer, name)
    raise AttributeError(name)
