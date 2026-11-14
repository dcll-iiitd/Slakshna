"""Build the two-node RAG test corpus from Wikipedia plain-text extracts."""
import json, re, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

OUT = Path(sys.argv[1])
NODES = {
    "node1": ["Chandrayaan-3", "Chandrayaan-2", "Mars Orbiter Mission", "Aditya-L1", "Gaganyaan"],
    "node2": ["Vikram Sarabhai", "Satish Dhawan", "A. P. J. Abdul Kalam", "Homi J. Bhabha",
              "Kaziranga National Park", "Indus Valley Civilisation"],
}
DROP = {"See also", "References", "External links", "Notes", "Further reading", "Bibliography",
        "Citations", "Sources", "Footnotes", "Explanatory notes"}
UA = "SlakshnaRAGCorpus/0.1 (academic research test corpus)"

font = "Helvetica"
for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
    if Path(p).exists():
        pdfmetrics.registerFont(TTFont("DejaVu", p)); font = "DejaVu"

def fetch(title):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "prop": "extracts|info", "explaintext": 1, "format": "json",
        "redirects": 1, "inprop": "url", "titles": title})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            data = json.load(urllib.request.urlopen(req, timeout=30)); break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5: raise
            time.sleep(20 * (attempt + 1))
    page = next(iter(data["query"]["pages"].values()))
    return page["title"], page["extract"], page["fullurl"], page["lastrevid"]

def sections(text):
    """Yield (level, heading, body) with boilerplate sections removed."""
    parts = re.split(r"^(={2,})\s*(.+?)\s*=+\s*$", text, flags=re.M)
    yield 0, None, parts[0]
    skip_level = None
    for i in range(1, len(parts), 3):
        level, head, body = len(parts[i]), parts[i + 1], parts[i + 2]
        if skip_level and level > skip_level: continue
        skip_level = None
        if head in DROP: skip_level = level; continue
        yield level, head, body

styles = getSampleStyleSheet()
for s in styles.byName.values(): s.fontName = font
manifest = []
for node, titles in NODES.items():
    (OUT / node).mkdir(parents=True, exist_ok=True)
    for t in titles:
        title, text, url, rev = fetch(t); time.sleep(4)
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        flow = [Paragraph(escape(title), styles["Title"]),
                Paragraph(escape(f"Source: {url} (revision {rev}). Text from Wikipedia, licensed CC BY-SA 4.0."), styles["Italic"]),
                Spacer(1, 12)]
        plain = [title, ""]
        for level, head, body in sections(text):
            if head:
                flow.append(Paragraph(escape(head), styles["Heading2" if level == 2 else "Heading3"]))
                plain += [head, ""]
            for para in body.split("\n"):
                para = para.strip()
                if para:
                    flow.append(Paragraph(escape(para), styles["BodyText"])); plain.append(para)
        pdf = OUT / node / f"{slug}.pdf"
        SimpleDocTemplate(str(pdf), pagesize=A4, title=title, author="Wikipedia contributors").build(flow)
        words = sum(len(p.split()) for p in plain)
        manifest.append({"node": node, "file": f"{node}/{slug}.pdf", "title": title, "url": url,
                         "revision": rev, "words": words})
        print(node, title, words, "words", file=sys.stderr)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
