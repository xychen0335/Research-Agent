#!/usr/bin/env python
from research_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare", *__import__("sys").argv[1:]]))
