import argparse
import webbrowser


def main() -> None:
    p = argparse.ArgumentParser(prog="hyppo gui")
    p.add_argument("--db", default="hyppo_gui.db")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    import uvicorn
    from hyppo.gui.app import create_app

    app = create_app(db_path=args.db)
    if not args.no_browser:
        webbrowser.open(f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
