from django import template
from django.shortcuts import render, get_object_or_404
from django.utils.translation import gettext_lazy as _, get_language


register = template.Library()


@register.simple_tag
def amos_name():
    return 'Amos Baranes'

#
# @register.filter
# def model_name(obj):
#     try:
#         return obj._meta.model_name
#     except AttributeError:
#         return None


@register.filter
def model_name(obj):
    try:
        return obj._meta.model_name
    except AttributeError:
        return None


@register.filter(name='has_group')
def has_group(user, group_name):
    return user.groups.filter(name=group_name).exists()


@register.filter(name='has_registered_course')
def has_registered_course(user, course_name):
    result = False
    registered_courses = user.course_schedule_users.all()
    for csu in registered_courses:
        print(csu.course_schedule.course.name)
        result = csu.course_schedule.course.name == course_name
        print(result)
    return result


@register.filter(name='times')
def times(number):
    return range(number)


@register.filter
def lang(obj):
    return obj.objects.filter(translations__language_code=get_language())[0]


@register.filter('get_value_from_dict')
def get_value_from_dict(dict_data, key):
    """
    usage example {{ your_dict|get_value_from_dict:your_key }}
    """
    if key:
        return dict_data.get(key)

#
# @register.inclusion_tag('blog/post/latest_posts.html')
# def show_latest_posts(count=5):
#     latest_posts = Post.published.order_by('-publish')[:count]
#     return {'latest_posts': latest_posts}



