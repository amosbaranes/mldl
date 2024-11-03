from django.apps import AppConfig

#
# class CoreConfig(AppConfig):
#     name = 'core'

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'academycity.apps.core'  # Use the full path
