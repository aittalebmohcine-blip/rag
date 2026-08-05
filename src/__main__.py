from fire import Fire

import sys
import json

from .CLI import CLI


def main() -> None:
    Fire(CLI)


if __name__ == "__main__":
    try:
        main()

    except json.decoder.JSONDecodeError as e:
        print(f"\nInvalid JSON: {e}", file=sys.stderr)
        raise SystemExit(1)

    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        raise SystemExit(1)

    except (
            FileNotFoundError,
            IsADirectoryError,
            PermissionError
    ) as e:
        print(f"\n{e}", file=sys.stderr)
        raise SystemExit(1)

    except Exception as e:
        print(f"\nUnexpected Error: {e}", file=sys.stderr)
        raise SystemExit(1)
