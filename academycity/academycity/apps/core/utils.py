from .models import Numbers, Debug
from datetime import datetime, date
import calendar
from dateutil.relativedelta import relativedelta


def add_years_months_days(sourcedate, years, months, days):
    new_date = sourcedate + relativedelta(years=+years)
    new_date = new_date + relativedelta(months=+months)
    new_date = new_date + relativedelta(days=+days)
    return new_date
    # day = days
    # month = sourcedate.month - 1 + months
    # year = sourcedate.year + years + month // 12
    # month = month % 12 + 1
    # day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    # year_date = date(year=year, month=month, day=day)
    # x = (year_date + pd.DateOffset(days=days)).to_pydate()
    # print(x)
    # return x


def get_number(source):
    obj, created = Numbers.objects.get_or_create(source=source, defaults={'number': 0})
    obj.number += 1
    obj.save()
    return obj.number


def log_debug(value):
    Debug.objects.create(value=value)


def clear_log_debug():
    Debug.truncate()


def is_role(user, role):
    return user.groups.filter(name=role).exists()



# Geo info
# https://docs.djangoproject.com/en/3.0/ref/contrib/gis/tutorial/
# https://stackoverflow.com/questions/2218093/django-retrieve-ip-location