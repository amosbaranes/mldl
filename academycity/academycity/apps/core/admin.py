from django.contrib import admin
from django.utils.translation import gettext_lazy as _

from .models import (Debug, DataAdvancedTabs, DataAdvancedTabsManager, DataAdvancedApps, Adjectives, AdjectivesValues,
                     GeneralData)


@admin.register(Debug)
class DebugAdmin(admin.ModelAdmin):
    list_display = ['id', 'value']


@admin.register(DataAdvancedTabs)
class DataAdvancedTabsAdmin(admin.ModelAdmin):
    list_display = ['manager', 'id', 'tab_name', 'order']


@admin.register(DataAdvancedApps)
class DataAdvancedAppsAdmin(admin.ModelAdmin):
    list_display = ['id', 'app_name']


@admin.register(DataAdvancedTabsManager)
class DataAdvancedTabsManagerAdmin(admin.ModelAdmin):
    list_display = ['id', 'at_name']


@admin.register(Adjectives)
class AdjectivesAdmin(admin.ModelAdmin):
    list_display = ['id', 'title']


@admin.register(AdjectivesValues)
class AdjectivesValuesAdmin(admin.ModelAdmin):
    list_display = ['id', 'adjective', 'order', 'value']


@admin.register(GeneralData)
class GeneralDataAdmin(admin.ModelAdmin):
    list_display = ['id', 'app', 'group', 'data_name']

