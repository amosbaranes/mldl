from __future__ import unicode_literals
from django.db import models
from academycity.apps.core.sql import TruncateTableMixin


class MLWeb(TruncateTableMixin, models.Model):
    company_name = models.CharField(max_length=100, default='', blank=True, null=True)
    address = models.CharField(max_length=50, default='', blank=True, null=True)

    def __str__(self):
        return self.company_name

