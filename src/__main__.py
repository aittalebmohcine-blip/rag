from fire import Fire

import sys
import json

from src.CLI import CLI


def main() -> None:
    try:
        Fire(CLI)


    except json.decoder.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        raise SystemExit(1)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)

    except (
            FileNotFoundError,
            IsADirectoryError,
            PermissionError
    ) as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)

    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
