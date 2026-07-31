"""External callback for Isaac Lab command-line task registration."""


def register_tasks():
    """Register wheeled-legged Gym environments for Isaac Lab CLI entrypoints."""
    import wheeled_legged_rl.tasks.velocity  # noqa: F401

    import os

    if os.environ.get("WHEELED_RL_DIAGNOSTIC_LOG"):
        from scripts.play_rsl_rl_diagnostic import install_diagnostic_hook

        install_diagnostic_hook()

    return None
