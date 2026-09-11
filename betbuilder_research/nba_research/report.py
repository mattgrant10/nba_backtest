"""Portable PDF exports; figures remain available at full resolution separately."""

from pathlib import Path
from io import BytesIO
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Preformatted
from PIL import Image as PILImage


def write_report(path, figures, audit, log_file=None):
    page = landscape(A4)
    styles = getSampleStyleSheet()
    styles["Code"].fontSize = 5
    styles["Code"].leading = 6
    doc = SimpleDocTemplate(
        str(path), pagesize=page, rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28
    )
    story = [
        Paragraph("NBA player statistics: descriptive research", styles["Title"]),
        Spacer(1, 16),
        Paragraph(
            f"{escape(audit['date_start'])} to {escape(audit['date_end'])}", styles["Normal"]
        ),
        Paragraph(
            f"{audit['rows']:,} source player-games; {audit['excluded_under_five_minutes']:,} excluded by the minutes filter (including missing minutes).",
            styles["Normal"],
        ),
        Spacer(1, 18),
        Paragraph("Interpretation and limitations", styles["Heading2"]),
        Paragraph(
            "Historical summaries and fixed-template replay are in-sample. They do not establish forecast accuracy or profitable betting. Full-sample averages and name matching are preserved for numerical comparison.",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph(
            "Legacy tail measures are correlations conditional on the first variable being in a tail, not copula tail-dependence coefficients. Family surfaces are illustrative approximations, not validated fitted copula densities. Distribution fit rankings retain known test limitations. See docs/ANALYTICAL_REVIEW.md.",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph(
            "Each figure is also exported as a PNG. All numerical result tables and source provenance accompany this PDF.",
            styles["Normal"],
        ),
    ]
    if log_file:
        import textwrap

        lines = Path(log_file).read_text().splitlines()
        # Keep the original grid tables, wrapping overwide lines so nothing is cropped.
        wrapped = [
            part
            for line in lines
            for part in (
                textwrap.wrap(line, width=205, replace_whitespace=False, drop_whitespace=False)
                or [""]
            )
        ]
        for start in range(0, len(wrapped), 78):
            story += [
                PageBreak(),
                Preformatted("\n".join(wrapped[start : start + 78]), styles["Code"]),
            ]
    for figure in figures:
        # Embed a bounded high-quality viewing copy; original PNG remains lossless.
        with PILImage.open(figure) as original:
            im = original.convert("RGB")
            im.thumbnail((2000, 1500), PILImage.Resampling.LANCZOS)
            width, height = im.size
            preview = BytesIO()
            im.save(preview, format="JPEG", quality=85, subsampling=0)
            preview.seek(0)
        scale = min((page[0] - 64) / width, (page[1] - 110) / height)
        story += [
            PageBreak(),
            Paragraph(escape(Path(figure).stem.replace("_", " ")), styles["Heading2"]),
            Image(preview, width=width * scale, height=height * scale),
        ]

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(28, 14, "Descriptive baseline | no validated betting profitability")
        canvas.drawRightString(page[0] - 28, 14, str(doc.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
