from __future__ import unicode_literals
from django.contrib import admin
from .models import (MLWeb)


@admin.register(MLWeb)
class MLWebAdmin(admin.ModelAdmin):
    list_display = ('id', 'company_name')

