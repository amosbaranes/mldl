# Generated by Django 3.1.13 on 2022-09-25 12:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0015_dataadvancedapps'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dataadvancedapps',
            name='app_name',
            field=models.CharField(default='app_name', max_length=50, null=True),
        ),
    ]
