from fastapi import FastAPI


def create_app(db_path: str = "hyppo_gui.db") -> FastAPI:
    app = FastAPI(title="Hyppo GUI")
    app.state.db_path = db_path

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
