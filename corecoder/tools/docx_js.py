"""Word document generation via docx-js (JavaScript).

Lets the agent write docx-js code directly to produce documents with
features python-docx can't do well: tables of contents, tables, images,
headers/footers, page numbers, multi-column layouts.

Requires Node.js + `npm install -g docx` on the server. Set
CORECODER_NODE_PATH to the global node_modules dir (e.g.
/opt/node/lib/node_modules) so `require('docx')` resolves.
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from .base import Tool

OUTPUT_DIR = Path(os.getenv("CORECODER_OUTPUT_DIR", str(Path.home() / ".corecoder" / "docs")))

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")

# JS template: docx library is imported, saveDoc(doc) writes the buffer to the
# target path. The agent's code runs inside main() and must call saveDoc(doc).
_TEMPLATE = r'''const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        Header, Footer, AlignmentType, PageOrientation, LevelFormat, ExternalHyperlink,
        InternalHyperlink, Bookmark, FootnoteReferenceRun, PositionalTab,
        PositionalTabAlignment, PositionalTabRelativeTo, PositionalTabLeader,
        TabStopType, TabStopPosition, Column, SectionType,
        TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType,
        VerticalAlign, PageNumber, PageBreak } = require("docx");

const OUTPUT_PATH = __OUTPUT_PATH__;

async function saveDoc(doc) {
  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync(OUTPUT_PATH, buf);
}

async function main() {
__AGENT_CODE__
}

main().catch(e => { console.error(e && e.stack || e); process.exit(1); });
'''


class WriteDocxJsTool(Tool):
    name = "write_docx_js"
    description = (
        "Generate a Word (.docx) document by writing docx-js JavaScript code. "
        "Use this for advanced formatting that write_docx can't do: tables of "
        "contents, tables, images, headers/footers, page numbers, multi-column. "
        "The docx library is already imported (Document, Packer, Paragraph, "
        "TextRun, Table, TableRow, TableCell, HeadingLevel, AlignmentType, "
        "LevelFormat, BorderStyle, WidthType, ShadingType, TableOfContents, "
        "Header, Footer, PageNumber, PageBreak, ImageRun, ExternalHyperlink, "
        "Bookmark, InternalHyperlink, FootnoteReferenceRun, Column, SectionType). "
        "Build a Document then call `await saveDoc(doc)`. "
        "Example:\n"
        "  const doc = new Document({\n"
        "    sections: [{ children: [\n"
        "      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(\"Report\")] }),\n"
        "      new Paragraph({ children: [new TextRun(\"Body text here.\")] }),\n"
        "    ]}]\n"
        "  });\n"
        "  await saveDoc(doc);"
    )
    parameters = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "File name without .docx extension (e.g. 'quarterly-report')",
            },
            "js_code": {
                "type": "string",
                "description": (
                    "docx-js JavaScript code. Build a Document and call "
                    "`await saveDoc(doc)` at the end. Do NOT require('docx') "
                    "yourself — it's already imported."
                ),
            },
        },
        "required": ["filename", "js_code"],
    }

    def execute(self, filename: str, js_code: str) -> str:
        safe_name = _SAFE_NAME.sub("-", filename).strip("-_") or "document"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"{safe_name}.docx"
        counter = 1
        while path.exists():
            path = OUTPUT_DIR / f"{safe_name}-{counter}.docx"
            counter += 1

        script = (
            _TEMPLATE
            .replace("__OUTPUT_PATH__", json.dumps(str(path)))
            .replace("__AGENT_CODE__", js_code)
        )

        # write the temp script under OUTPUT_DIR so node resolves docx via NODE_PATH
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", dir=OUTPUT_DIR, delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            script_path = f.name

        env = os.environ.copy()
        # read at call time, not import time — .env is loaded by Config.from_env()
        # which runs after this module is imported, so a module-level constant
        # would capture an empty value.
        node_path = os.getenv("CORECODER_NODE_PATH", "")
        if node_path:
            env["NODE_PATH"] = node_path

        try:
            result = subprocess.run(
                ["node", script_path],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            if result.returncode != 0:
                err = result.stderr.strip()[-800:]
                return f"Error generating docx (node exit {result.returncode}):\n{err}"
            if not path.exists():
                out = result.stdout.strip()[-400:]
                return (
                    "Error: script finished but saveDoc was never called. "
                    f"stdout: {out}"
                )
            return f"文档已生成。下载链接：/download/{path.name}"
        except subprocess.TimeoutExpired:
            return "Error: docx generation timed out (30s limit)"
        except FileNotFoundError:
            return "Error: node not installed. Install Node.js to use write_docx_js."
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
