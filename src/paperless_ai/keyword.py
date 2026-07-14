from keybert import KeyBERT

from documents.models import DocumentKeyword

kw_model = KeyBERT()


def extract_keywords(document):

    text = document.content or ""

    keywords = kw_model.extract_keywords(

        text,

        top_n=30,

        keyphrase_ngram_range=(1, 2),

    )

    keywords = [

        k

        for k, score in keywords

    ]

    DocumentKeyword.objects.update_or_create(

        document=document,

        defaults={

            "keywords": keywords,

        },

    )