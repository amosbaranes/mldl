# Generated by Django 3.1.13 on 2022-03-03 01:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0006_dataadvancedtabs_tab_title'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataadvancedtabs',
            name='tab_content',
            field=models.JSONField(null=True),
        ),
    ]
