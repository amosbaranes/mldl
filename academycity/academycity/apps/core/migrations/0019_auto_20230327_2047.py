# Generated by Django 3.1.13 on 2023-03-27 17:47

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0018_generaldata'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='generaldata',
            options={'ordering': ['app', 'group', 'data_name'], 'verbose_name': 'general_data', 'verbose_name_plural': 'general_datas'},
        ),
    ]