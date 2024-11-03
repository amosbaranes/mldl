from django.shortcuts import render
from ...core.utils import log_debug
from django.urls import reverse
from ...core.apps_general_functions import activate_obj_function
from .models import MLWeb


def home(request):
    log_debug("home: ")

    company_obj_id_ = 1
    app_ = "ml"
    activate_obj_function_link_ = reverse(app_+':activate_obj_function', kwargs={})
    return render(request, app_+'/home.html', {"atm_name": "ml_default_tm", "app": app_,
                                               "app_activate_function_link": activate_obj_function_link_,
                                               "company_obj_id": company_obj_id_, "title": "Deep Learning"}
                  )


def app_id(request, app_name, company_obj_id):
    log_debug("app_id: ")
    company_obj = MLWeb.objects.get_or_create(id=company_obj_id)
    app_ = "ml"
    app_activate_function_link_ = reverse(app_+':activate_obj_function', kwargs={})
    return render(request, app_+'/home.html', {"atm_name": "ml_"+app_name+"_tm", "app": app_,
                                               "app_activate_function_link": app_activate_function_link_,
                                               "company_obj_id": company_obj_id, "title": "Deep Learning"}
                  )


def app(request, app_name):
    log_debug("app: ")
    return app_id(request, app_name, 1)

