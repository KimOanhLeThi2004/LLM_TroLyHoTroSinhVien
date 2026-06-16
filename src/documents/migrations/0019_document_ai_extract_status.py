# Generated for Paperless AI extraction status integration

from django.db import migrations
from django.db import models


class Migration(migrations.Migration):
    dependencies = [
        ("documents", "0018_saved_view_simple_search_rules"),
    ]

    operations = [
        migrations.AddField(
            model_name="document",
            name="ai_extract_status",
            field=models.CharField(
                choices=[
                    ("queued", "Queued"),
                    ("processing", "Processing"),
                    ("gemini_processing", "Gemini processing"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                ],
                db_index=True,
                default="completed",
                help_text="Current AI extraction lifecycle status for this document.",
                max_length=32,
                verbose_name="AI extract status",
            ),
        ),
        migrations.AddField(
            model_name="document",
            name="ai_extract_reason",
            field=models.TextField(
                blank=True,
                help_text="Human-readable reason for the current AI extraction status.",
                verbose_name="AI extract reason",
            ),
        ),
        migrations.AddField(
            model_name="document",
            name="ai_extract_mode",
            field=models.CharField(
                blank=True,
                help_text="Extraction mode selected for this document.",
                max_length=64,
                verbose_name="AI extract mode",
            ),
        ),
        migrations.AddField(
            model_name="document",
            name="ai_extract_score",
            field=models.FloatField(
                blank=True,
                help_text="Latest OCR quality score used by the AI extraction pipeline.",
                null=True,
                verbose_name="AI extract score",
            ),
        ),
        migrations.AddField(
            model_name="document",
            name="ai_extract_error",
            field=models.JSONField(
                blank=True,
                help_text="Latest AI extraction error payload, if extraction failed.",
                null=True,
                verbose_name="AI extract error",
            ),
        ),
    ]
