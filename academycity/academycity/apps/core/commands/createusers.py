# We can run this command with: python manage.py createusers
# https://simpleisbetterthancomplex.com/tutorial/2018/08/27/how-to-create-custom-django-management-commands.html


# check how to use it....
# -----------------------

from django.conf import settings

User = settings.AUTH_USER_MODEL

class Command(BaseCommand):
    help = 'Create Users'

    def handle(self, *args, **kwargs):
        User.objects.create_superuser(username='admin', email='amos@drbaranes.com', password='sql1pass')
        User.objects.create_user(username='student', email='amos@drbaranes.com', password='student')
        User.objects.create_user(username='instructor', email='amos@drbaranes.com', password='instructor')