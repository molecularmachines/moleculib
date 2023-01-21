import nox


@nox.session
def tests(session):
    session.install("pytest")
    session.run("pip", "install", ".", "-v")
    session.run("pytest", *session.posargs)
