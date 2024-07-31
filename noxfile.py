import nox


@nox.session(venv_backend="conda")
def tests(session):
    session.conda_install("pytest", "pandas", "dask", "numpy", "scipy", "pytables")
    session.install(".")
    session.run('pytest')
