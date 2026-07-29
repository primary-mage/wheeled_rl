"""External callback for Isaac Lab command-line task registration."""


def register_tasks():
    """Register wheeled-legged Gym environments for Isaac Lab CLI entrypoints."""
    import wheeled_legged_rl.tasks.velocity  # noqa: F401

    return None

