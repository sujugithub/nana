"""Downloadable outputs derived from completed, account-scoped chat messages.

No model-produced paths, code execution, uploads or extra inference are needed.
The saved reply is the source of truth, so reload and deletion follow chat.
"""
from io import BytesIO
import re
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

FENCES = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)\n(?P<code>[\s\S]*?)^ {0,3}(?P=fence)[ \t]*(?:\n|$)", re.M)
EXTENSIONS = {
    "html": "html", "htm": "html", "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts", "jsx": "jsx", "tsx": "tsx",
    "python": "py", "py": "py", "css": "css", "json": "json",
    "csv": "csv", "markdown": "md", "md": "md", "text": "txt",
    "txt": "txt", "plain": "txt", "sql": "sql", "bash": "sh",
    "sh": "sh", "shell": "sh", "yaml": "yaml", "yml": "yaml",
    "xml": "xml", "java": "java", "c": "c", "cpp": "cpp",
    "c++": "cpp", "rust": "rs", "go": "go", "swift": "swift",
}
MIMES = {"docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
         "html": "text/html; charset=utf-8", "json": "application/json; charset=utf-8",
         "csv": "text/csv; charset=utf-8", "md": "text/markdown; charset=utf-8"}


def _no_files(prompt):
    return bool(re.search(r"\b(?:no|without|don't|do not)\s+(?:a\s+)?(?:file|download)s?\b", prompt, re.I))


def requested_format(prompt):
    """Require an output request, rather than turning format questions into files."""
    lower = prompt.lower()
    if _no_files(prompt):
        return None
    if re.match(r"\s*(?:what\b|why\b|explain\b|how (?:do|does|can|to)\b)", lower) and not re.search(r"\b(?:export|download|save|convert)\b", lower):
        return None
    # Explicit conversion/output wording, including a follow-up 'in docx please'.
    output = re.search(r"\b(?:write|wrike|make|create|generate|build|export|download|save|convert|give|provide)\b", lower)
    followup = re.search(r"\b(?:as|in|to)\s+(?:a\s+)?(?:\.?(?:docx|html|txt|md|csv|json)|word (?:document|file))\b", lower)
    if not (output or followup):
        return None
    if re.search(r"\b(?:what|why|how)\b.*\b(?:docx|html|txt|md|csv|json)\b", lower) and not output:
        return None
    for ext in ("docx", "html", "csv", "json", "md", "txt"):
        # A filename alone can describe an input attachment, not the desired output.
        standalone = re.search(r"(?<![\w.])\.?" + ext + r"\b", lower)
        named_output = re.search(r"\b(?:save|export|download|create|generate|build|make|named?|called)\b[^\n]{0,60}\b[\w-]+\." + ext + r"\b", lower) or re.search(r"\b(?:as|in|to)\s+[\w-]+\." + ext + r"\b", lower)
        if standalone or named_output:
            return ext
    if re.search(r"\bword (?:document|file)\b", lower):
        return "docx"
    return None


def instructions(prompt):
    base = ("For code, use complete fenced code blocks with a language tag. "
            "Chat adds Copy and Download controls to closed code blocks. "
            "For a browser game with no requested language, prefer one self-contained HTML file. "
            "Do not invent download URLs or claim a file was uploaded or executed. ")
    ext = requested_format(prompt)
    if ext == "docx":
        base += ("The user requests a Word file. Write the finished document in Markdown "
                 "with a descriptive title, headings and paragraphs, without packaging instructions. "
                 "Nana packages your text locally into a real DOCX download; no special model capability is needed. ")
    elif ext:
        base += f"The user requests {ext.upper()} output. Provide the complete file content in a {ext} fenced block. "
        if ext == "html":
            base += "For a browser game, include HTML, CSS and JavaScript together so the downloaded file runs on its own. "
    return base


def _name(prompt, ext, index=0, info=""):
    # Honour a safe requested basename, never a filesystem path.
    named = re.search(r"(?<![\w/\\])([A-Za-z0-9][A-Za-z0-9_.-]{0,70})\." + re.escape(ext) + r"\b", info + " " + prompt, re.I)
    stem = named.group(1) if named else "snake-game" if re.search(r"\bsnake\b", prompt, re.I) else "document" if ext == "docx" else "response" if ext in {"md", "txt"} else "code"
    stem = re.sub(r"[^A-Za-z0-9_-]", "-", stem).strip("-")[:70] or "output"
    return f"{stem}{'-' + str(index + 1) if index and not named else ''}.{ext}"


def _outputs(prompt, answer):
    if not answer.strip() or answer.startswith("[mock-") or _no_files(prompt):
        return []
    blocks = list(FENCES.finditer(answer))
    outputs = []
    for i, block in enumerate(blocks):
        code = block['code']
        if not code.strip():
            continue
        info = block['info'].strip()
        language = info.split()[0].lower() if info else "text"
        ext = EXTENSIONS.get(language, "txt")
        name = _name(prompt, ext, i, info)
        if any(output['name'] == name for output in outputs):
            name = name.rsplit('.', 1)[0] + f'-{i + 1}.' + ext
        outputs.append({"name": name, "mime": MIMES.get(ext, "text/plain; charset=utf-8"),
                        "kind": "code", "language": language, "code_index": i, "text": code})
    requested = requested_format(prompt)
    if requested == "docx":
        if re.match(r"\s*(?:sorry[,!]?\s*)?I (?:can(?:not|'t)|am unable to)\b", answer, re.I):
            return outputs  # Do not present a refusal as the requested document.
        # A full Markdown document in a single fence is also accepted.
        text = blocks[0]['code'] if len(blocks) == 1 and blocks[0]['info'].strip().lower() in {"markdown", "md", "text", ""} and not answer[blocks[0].end():].strip() else answer
        if text != answer:
            outputs = []
        outputs.insert(0, {"name": _name(prompt, "docx"), "mime": MIMES['docx'], "kind": "document", "text": text})
    elif requested in {"txt", "md"} and not outputs:
        outputs.append({"name": _name(prompt, requested), "mime": MIMES.get(requested, "text/plain; charset=utf-8"), "kind": "document", "text": answer})
    elif not outputs and requested == "html" and re.match(r"\s*(?:<!doctype html[^>]*>\s*)?<html\b", answer, re.I) and re.search(r"</html>\s*$", answer, re.I):
        outputs.append({"name": _name(prompt, "html"), "mime": MIMES['html'], "kind": "code", "text": answer, "language": "html"})
    return outputs


def describe(prompt, answer):
    return [{**{k: v for k, v in output.items() if k != "text"}, "index": i}
            for i, output in enumerate(_outputs(prompt, answer))]


def _runs(text):
    runs = []
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        props = ""
        if part.startswith("**") and part.endswith("**"):
            props, part = "<w:b/>", part[2:-2]
        elif part.startswith("`") and part.endswith("`"):
            props, part = '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/>', part[1:-1]
        # XML 1.0 forbids most control characters, including model-generated NUL.
        part = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]", "", part)
        runs.append(f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return "".join(runs)


def make_docx(text):
    """Small, portable OOXML writer for essays, headings, lists and code blocks."""
    paragraphs, first, in_code = [], True, False
    for line in text.splitlines():
        if re.match(r"^\s*(?:`{3,}|~{3,})", line):
            in_code = not in_code
            continue
        style = "Normal"
        heading = re.match(r"^(#{1,6})\s+(.+)$", line) if not in_code else None
        if heading:
            level, line = len(heading[1]), heading[2]
            style = "Title" if first and level == 1 else "Heading" + str(min(level, 3))
        elif in_code:
            style = "Code"
        elif re.match(r"^\s*[-*+]\s+", line):
            line = "• " + re.sub(r"^\s*[-*+]\s+", "", line)
        if line.strip():
            first = False
        paragraphs.append(f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{_runs(line)}</w:p>')
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
                + "".join(paragraphs) + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
                '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body></w:document>')
    styles = ('<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
              '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>'
              '<w:sz w:val="24"/><w:color w:val="000000"/></w:rPr></w:rPrDefault>'
              '<w:pPrDefault><w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>'
              '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>')
    for name, size in [("Title", 36), ("Heading1", 30), ("Heading2", 27), ("Heading3", 25), ("Code", 20)]:
        font = '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/>' if name == "Code" else '<w:b/>'
        keep = "" if name == "Code" else "<w:keepNext/>"
        styles += (f'<w:style w:type="paragraph" w:styleId="{name}"><w:name w:val="{name}"/><w:basedOn w:val="Normal"/>'
                   f'<w:pPr>{keep}<w:spacing w:before="200" w:after="100"/></w:pPr>'
                   f'<w:rPr>{font}<w:sz w:val="{size}"/><w:color w:val="000000"/></w:rPr></w:style>')
    styles += '</w:styles>'
    content_types = ('<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                     '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
    rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    doc_rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
    out = BytesIO()
    with ZipFile(out, 'w', ZIP_DEFLATED) as archive:
        for name, content in {'[Content_Types].xml': content_types, '_rels/.rels': rels,
                              'word/document.xml': document, 'word/styles.xml': styles,
                              'word/_rels/document.xml.rels': doc_rels}.items():
            archive.writestr(name, content.encode('utf-8'))
    return out.getvalue()


def download(conversation_id, message_id, index):
    from .database import get_connection
    conn = get_connection()  # Resolves the signed-in account's database.
    try:
        row = conn.execute("SELECT * FROM messages WHERE id=? AND conversation_id=? AND role='assistant'", (message_id, conversation_id)).fetchone()
        if row is None:
            raise LookupError("Reply not found")
        user = conn.execute("SELECT content FROM messages WHERE conversation_id=? AND role='user' AND id<? ORDER BY id DESC LIMIT 1", (conversation_id, message_id)).fetchone()
        outputs = _outputs(user['content'] if user else '', row['content'])
        if index < 0 or index >= len(outputs):
            raise LookupError("File not found")
        output = outputs[index]
        raw = make_docx(output['text']) if output['name'].endswith('.docx') else output['text'].encode('utf-8')
        return raw, output['mime'], output['name']
    finally:
        conn.close()
