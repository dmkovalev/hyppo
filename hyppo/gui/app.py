import logging
from pathlib import Path

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def create_app(
    db_path: str = "hyppo_gui.db", static_dir: Path | None = None
) -> FastAPI:
    app = FastAPI(title="Hyppo GUI")
    app.state.db_path = db_path

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    from hyppo.gui.api import projects as projects_api
    from hyppo.gui.projects import ProjectStore

    app.state.projects = ProjectStore(db_path=db_path)
    from hyppo.gui.demo import seed_demo

    seed_demo(app.state.projects)
    app.include_router(projects_api.router)

    from hyppo.gui.api import hypotheses as hypotheses_api

    app.include_router(hypotheses_api.router)

    from hyppo.gui.api import ve as ve_api

    app.include_router(ve_api.router)

    from hyppo.gui.api import real as real_api

    app.include_router(real_api.router)

    from hyppo.gui.api import graph as graph_api
    from hyppo.gui.api import plan as plan_api

    app.include_router(graph_api.router)
    app.include_router(plan_api.router)

    from hyppo.gui.api import runs as runs_api

    app.include_router(runs_api.router)

    from hyppo.gui.api import comparison as comparison_api

    app.include_router(comparison_api.router)

    from hyppo.gui import ws as ws_module

    app.include_router(ws_module.router)

    from fastapi.staticfiles import StaticFiles

    if static_dir is None:
        static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="spa")
    else:
        logger.warning(
            "SPA static assets not found at %s; run `cd webui && npm ci && "
            "npm run build` then copy webui/dist/* into hyppo/gui/static/ "
            "to enable the web UI (API-only mode active)",
            static_dir,
        )

    return app
