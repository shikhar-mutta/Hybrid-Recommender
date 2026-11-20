#!/usr/bin/env bash
# Usage: ./run.sh [test|notebook|report|all|web-data|api|web]
set -e
cd "$(dirname "$0")"
[ -d data/ml-latest-small ] || { mkdir -p data && curl -sL -o data/ml.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip && unzip -qo data/ml.zip -d data; }
case "${1:-all}" in
  test)     python3 tests/test_project.py ;;
  notebook) python3 build/make_full_nb.py && python3 build/exec_nb.py ;;
  report)   d=$(mktemp -d) && cp -r report/report.tex report/demo_section.tex report/tables report/figs "$d"/ && (cd "$d" && latexmk -pdf -interaction=nonstopmode report.tex >/dev/null) && cp "$d/report.pdf" report/report.pdf && echo "report/report.pdf built" ;;
  web-data) python3 build/export_web_data.py ;;                       # (re)export JSON for the web app
  api)      python3 server/api.py ;;                                  # live API on :8000 (lambda slider, new-film demo)
  web)      cd web && { [ -d node_modules ] || npm install; } && npm run dev -- --host ;;   # UI on http://localhost:5173
  all)      "$0" test && "$0" notebook && "$0" report ;;
esac
