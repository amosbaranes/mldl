from django.db import models
from django.db import connection
from .fields import OrderField
from .sql import TruncateTableMixin


class DataAdvancedTabsManager(models.Model):
    at_name = models.CharField(max_length=50, null=True, default="new")
    manager_content = models.JSONField(null=True)

    def __str__(self):
        return str(self.at_name)


class DataAdvancedTabs(models.Model):
    class Meta:
        verbose_name = 'DataAdvancedTabs'
        verbose_name_plural = 'DataAdvancedTabs'
        ordering = ['order']

    manager = models.ForeignKey(DataAdvancedTabsManager, on_delete=models.CASCADE, default=1)
    tab_name = models.CharField(max_length=50, null=True, default="tab")
    tab_content = models.JSONField(null=True)
    order = models.SmallIntegerField(blank=True, default=1)

    def __str__(self):
        return str(self.manager) + ":" + str(self.tab_name)


class DataAdvancedApps(models.Model):
    class Meta:
        verbose_name = 'DataAdvancedApps'
        verbose_name_plural = 'DataAdvancedApps'

    app_name = models.CharField(max_length=50, null=True, default="app_name")
    app_content = models.JSONField(null=True)

    def __str__(self):
        return str(self.app_name)


class DataAdvancedTabsCustomization(models.Model):
    tab = models.ForeignKey(DataAdvancedTabs, on_delete=models.CASCADE, default=1)
    tab_content = models.JSONField(null=True)

    def __str__(self):
        return str(self.manager) + ":" + str(self.tab_name)


class ModifyModel(object):

    @classmethod
    def truncate(cls):
        with connection.cursor() as cursor:
            cursor.execute('TRUNCATE TABLE {} RESTART IDENTITY CASCADE'.format(cls._meta.db_table))


class Adjectives(models.Model):
    title = models.CharField(max_length=50)

    def __str__(self):
        return str(self.title)


class AdjectiveQuerySet(models.QuerySet):
    def adjectives(self, adjective_title):
        return self.filter(adjective__title=adjective_title)


class AdjectiveManager(models.Manager):
    def get_queryset(self):
        return AdjectiveQuerySet(self.model, self._db)

    def adjectives(self, adjective_title):
        # print(adjective_title, "\n", self.get_queryset(), "\n")
        # print(adjective_title, "\n", self.get_queryset().adjectives(adjective_title), "\n")
        return self.get_queryset().adjectives(adjective_title).order_by("order")

#
# class AdjectiveManager(models.Manager):
#     def get_queryset(self, adjective_title):
#         return super(AdjectiveManager, self).get_queryset().filter(adjective__title=adjective_title).orderby(self.order)


class AdjectivesValues(models.Model):
    adjective = models.ForeignKey(Adjectives, on_delete=models.CASCADE)
    order = models.SmallIntegerField(blank=True, default=1)
    value = models.CharField(max_length=50)

    objects = models.Manager()  # The default manager.
    adjectives = AdjectiveManager()  # Our custom manager.

    def __str__(self):
        return self.value


class Numbers(models.Model):
    SOURCES = (
        (0, 'Other'),
        (1, 'General Ledger'),
    )
    source = models.IntegerField(default=1, choices=SOURCES)
    number = models.PositiveIntegerField()

    def __str__(self):
        return str(self.source) + str(self.number)


class Debug(TruncateTableMixin, models.Model):
    value = models.CharField(max_length=1024)

    def __str__(self):
        return self.value


class GeneralData(TruncateTableMixin, models.Model):
    class Meta:
        verbose_name = 'general_data'
        verbose_name_plural = 'general_datas'
        ordering = ['app', 'group', 'data_name']
    app = models.CharField(max_length=40, default='', blank=True, null=True)
    group = models.CharField(max_length=40, default='general', blank=True, null=True)
    data_name = models.CharField(max_length=40, default='', blank=True, null=True)
    data_json = models.JSONField(null=True)

    def __str__(self):
        return str(self.data_name)
