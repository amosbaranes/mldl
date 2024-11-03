# Generated by Django 3.1.13 on 2022-11-0  08:36

import academycity.apps.core.sql
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MLWeb',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('company_name', models.CharField(blank=True, default='', max_length=100, null=True)),
                ('address', models.CharField(blank=True, default='', max_length=50, null=True)),
            ],
            bases=(academycity.apps.core.sql.TruncateTableMixin, models.Model),
        ),
    ]
