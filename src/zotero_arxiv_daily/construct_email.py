from .protocol import Paper
import math


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em;
      line-height: 1;
      display: inline-flex;
      align-items: center;
    }
    .half-star {
      display: inline-block;
      width: 0.5em;
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""


def get_empty_html():
    return """
  <table border="0" cellpadding="0" cellspacing="0" width="100%"
   style="font-family: Arial, sans-serif; border: 1px solid #ddd;
          border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr><td style="font-size: 20px; font-weight: bold; color: #333;">
      No Papers Today. Take a Rest!
  </td></tr>
  </table>
  """


def _format_tldr(tldr: str) -> str:
    """Convert structured 5-point TLDR text to simple HTML paragraphs."""
    if not tldr:
        return ""
    labels = [
        "Background and context:",
        "Problem addressed:",
        "Methods:",
        "Conclusions:",
        "Open problems:",
    ]
    # Bold the section labels, wrap each section in a paragraph
    result = tldr
    for label in labels:
        result = result.replace(label, f"<br><strong>{label}</strong> ")
    return result.lstrip("<br>")


def get_block_html(title, authors, rate, tldr, pdf_url, affiliations=None,
                   watchlist_hit=None, llm_reason=None):
    if watchlist_hit:
        label = "📌 WATCHLIST — {wtype}: {matched}".format(
            wtype=watchlist_hit['type'], matched=watchlist_hit['matched'])
        watchlist_row = (
            '\n    <tr><td style="padding: 4px 0;">'
            '<span style="background-color:#c0392b;color:white;padding:3px 8px;'
            'border-radius:4px;font-size:13px;font-weight:bold;">{label}</span>'
            '</td></tr>'
        ).format(label=label)
    else:
        watchlist_row = ""

    if llm_reason:
        llm_reason_row = (
            '\n    <tr><td style="font-size:13px;color:#777;padding:4px 0;font-style:italic;">'
            '💡 {reason}</td></tr>'
        ).format(reason=llm_reason)
    else:
        llm_reason_row = ""

    block = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%"
     style="font-family: Arial, sans-serif; border: 1px solid #ddd;
            border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr><td style="font-size: 20px; font-weight: bold; color: #333;">{title}</td></tr>
    {watchlist_row}
    <tr><td style="font-size: 14px; color: #666; padding: 8px 0;">
        {authors}<br><i>{affiliations}</i>
    </td></tr>
    <tr><td style="font-size: 14px; color: #333; padding: 8px 0;">
        <strong>Relevance:</strong> {rate}
    </td></tr>
    {llm_reason_row}
    <tr><td style="font-size: 14px; color: #333; padding: 8px 0;">
        <strong>Summary:</strong><br>{tldr_html}
    </td></tr>
    <tr><td style="padding: 8px 0;">
        <a href="{pdf_url}" style="display:inline-block;text-decoration:none;
           font-size:14px;font-weight:bold;color:#fff;background-color:#d9534f;
           padding:8px 16px;border-radius:4px;">PDF</a>
    </td></tr>
    </table>
"""
    return block.format(
        title=title, authors=authors, rate=rate,
        tldr_html=_format_tldr(tldr),
        pdf_url=pdf_url, affiliations=affiliations or 'Unknown Affiliation',
        watchlist_row=watchlist_row, llm_reason_row=llm_reason_row,
    )


def get_classic_block_html(title, authors, citations, tldr, pdf_url, reason=""):
    """Render a card for a classic/foundational paper pick."""
    authors_str = authors if isinstance(authors, str) else ', '.join(authors)
    reason_row = ""
    if reason:
        reason_row = (
            '<tr><td style="font-size:13px;color:#333;padding:8px 0;">'
            '<strong>Summary:</strong><br>{}</td></tr>'
        ).format(_format_tldr(reason))

    return """
    <table border="0" cellpadding="0" cellspacing="0" width="100%"
     style="font-family: Arial, sans-serif; border: 1px solid #b8860b;
            border-radius: 8px; padding: 16px; background-color: #fffdf0;">
    <tr><td style="font-size: 18px; font-weight: bold; color: #333;">{title}</td></tr>
    <tr><td style="font-size: 13px; color: #666; padding: 4px 0;">
        {authors} &nbsp;·&nbsp; <strong>{citations}</strong> citations
    </td></tr>
    {reason_row}
    <tr><td style="padding: 8px 0;">
        <a href="{pdf_url}" style="display:inline-block;text-decoration:none;
           font-size:14px;font-weight:bold;color:#fff;background-color:#b8860b;
           padding:8px 16px;border-radius:4px;">PDF</a>
    </td></tr>
    </table>
""".format(title=title, authors=authors_str, citations=citations,
           reason_row=reason_row, pdf_url=pdf_url)


def get_section_header(title: str, subtitle: str = "") -> str:
    sub = f'<div style="font-size:13px;color:#888;margin-top:4px;">{subtitle}</div>' if subtitle else ""
    return """
<div style="font-family:Arial,sans-serif;border-bottom:2px solid #333;
            padding:12px 0 6px 0;margin:24px 0 12px 0;">
  <span style="font-size:20px;font-weight:bold;color:#222;">{title}</span>
  {sub}
</div>
""".format(title=title, sub=sub)


def get_stars(score: float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low, high = 6, 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high - low) / 10
        star_num = math.ceil((score - low) / interval)
        full_star_num = int(star_num / 2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">' + full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers: list[Paper], classic_papers: list[Paper] = None) -> str:
    parts = []

    # ── Today's new papers ───────────────────────────────────────────────────
    if not papers and not classic_papers:
        return framework.replace('__CONTENT__', get_empty_html())

    if papers:
        parts.append(get_section_header(
            "📄 Today's New Papers",
            "Ranked by LLM relevance · 📌 = watchlist author/affiliation"
        ))
        for p in papers:
            rate = '📌 pinned' if getattr(p, 'watchlist_hit', None) else (
                round(p.score, 1) if p.score is not None else 'Unknown'
            )
            author_list = p.authors
            if len(author_list) <= 5:
                authors = ', '.join(author_list)
            else:
                authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
            if p.affiliations is not None:
                affiliations = ', '.join(p.affiliations[:5])
                if len(p.affiliations) > 5:
                    affiliations += ', ...'
            else:
                affiliations = 'Unknown Affiliation'

            parts.append(get_block_html(
                p.title, authors, rate, p.tldr, p.pdf_url, affiliations,
                watchlist_hit=getattr(p, 'watchlist_hit', None),
                llm_reason=getattr(p, 'llm_reason', None),
            ))

    # ── Classic picks ────────────────────────────────────────────────────────
    if classic_papers:
        parts.append(get_section_header(
            "📚 Today's Classic Picks",
            "Foundational papers selected daily by LLM from highly-cited hep-th literature"
        ))
        for p in classic_papers:
            citations = getattr(p, 'citations', 0)
            reason = getattr(p, 'llm_reason', '') or p.tldr or ''
            parts.append(get_classic_block_html(
                p.title, p.authors, citations, p.tldr, p.pdf_url, reason=reason
            ))

    content = '<br>'.join(parts)
    return framework.replace('__CONTENT__', content)
