import re

from documents.models import DocumentLayout


def extract_layout(document):

    text = document.content or ""

    sections = []

    current = None

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if re.match(r"^\d+(\.\d+)*", line):

            current = {
                "title": line,
                "content": [],
            }

            sections.append(current)

        elif current:

            current["content"].append(line)

    DocumentLayout.objects.update_or_create(

        document=document,

        defaults={

            "layout": sections,

        },

    )