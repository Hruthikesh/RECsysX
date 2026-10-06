"""Copy the generated tables from results/tables/summary_tables.md into README.md / REPORT.md.

Each table in summary_tables.md starts with "<!-- name -->". In the docs a table is marked with
    <!-- table:name -->
    ...anything, replaced on every run...
    <!-- /table:name -->
so the numbers in the docs are always the ones the scripts produced.

    python scripts/update_docs.py
"""
import re

from recsysx.config import PROJECT_ROOT


def main():
    src = (PROJECT_ROOT / "results" / "tables" / "summary_tables.md").read_text(encoding="utf-8")
    parts = re.split(r"^<!-- (\w+) -->\n", src, flags=re.M)
    tables = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
    for doc in ("README.md", "REPORT.md"):
        path = PROJECT_ROOT / doc
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        used = []

        def repl(m):
            name = m.group(1)
            if name not in tables:
                raise KeyError(f"{doc}: no generated table called {name!r}")
            used.append(name)
            return f"<!-- table:{name} -->\n{tables[name]}\n<!-- /table:{name} -->"

        text = re.sub(r"<!-- table:(\w+) -->.*?<!-- /table:\1 -->", repl, text, flags=re.S)
        path.write_text(text, encoding="utf-8")
        print(f"{doc}: updated {used}")


if __name__ == "__main__":
    main()
