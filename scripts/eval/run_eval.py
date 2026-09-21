#!/usr/bin/env python
from research_agent.cli import main
import sys

if __name__ == "__main__":
    raise SystemExit(main(["eval", *sys.argv[1:]]))
