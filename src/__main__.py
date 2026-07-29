from fire import Fire

import sys

from src.CLI import CLI


def main() -> None:
    try:
        Fire(CLI)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)

    except PermissionError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
