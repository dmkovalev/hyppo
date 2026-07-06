from fastapi import FastAPI


def create_app(db_path: str = "hyppo_gui.db") -> FastAPI:
    app = FastAPI(title="Hyppo GUI")
    app.state.db_path = db_path

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    from hyppo.gui.projects import ProjectStore
    from hyppo.gui.api import projects as projects_api
    app.state.projects = ProjectStore(db_path=db_path)
    app.include_router(projects_api.router)

    from hyppo.gui.api import hypotheses as hypotheses_api
    app.include_router(hypotheses_api.router)

    return app
