from collections import Counter

from documents.models import DocumentKeyword
from documents.models import DocumentTopic


def classify_topic(document):

    try:

        keywords = DocumentKeyword.objects.get(

            document=document,

        ).keywords

    except DocumentKeyword.DoesNotExist:

        keywords = []

    if keywords:

        topic = Counter(keywords).most_common(1)[0][0]

        score = 1.0

    else:

        topic = "Unknown"

        score = 0

    DocumentTopic.objects.update_or_create(

        document=document,

        defaults={

            "topic": topic,

            "score": score,

        },

    )